#!/usr/bin/env python3
"""Run a leakage-conscious toxic-comment benchmark with exactly three models.

The script evaluates Logistic Regression, Linear SVM and a compact multilabel
MLP with exactly two feature-extraction methods: word TF-IDF and a binary
character bag of n-grams.
It is designed for local execution from VS Code on a CPU workstation and
deliberately performs no downloads or web searches. A test set and sample
submission are optional; when both are
provided, the strict default writes a best-single submission. A separately
audited rank ensemble is available only through an explicit opt-in.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import importlib.metadata
import inspect
import io
import json
import logging
import math
import os
import platform
import random
import re
import time
import unicodedata
import warnings
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC


LOGGER = logging.getLogger("toxic_comment_three_models")

LABEL_COLUMNS: tuple[str, ...] = (
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
)
TEXT_COLUMN = "comment_text"
ID_COLUMN = "id"

# These tuples are intentionally closed. Explicit validation later in the script
# prevents accidental expansion beyond the three algorithms and two feature-
# extraction methods required by the assessment.
MODEL_FAMILIES: tuple[str, ...] = (
    "Logistic Regression",
    "Linear SVM",
    "Multilayer Perceptron",
)
REPRESENTATIONS: tuple[str, ...] = (
    "Word TF-IDF",
    "Character Bag-of-N-grams",
)

MODEL_KEY = {
    "Logistic Regression": "logistic_regression",
    "Linear SVM": "linear_svm",
    "Multilayer Perceptron": "multilayer_perceptron",
}
REPRESENTATION_KEY = {
    "Word TF-IDF": "word_tfidf",
    "Character Bag-of-N-grams": "character_bag_of_ngrams",
}
MODEL_SELECTION_RATIONALE = (
    "The three families were frozen before the definitive run using only the "
    "supplied project artefacts and local CPU pilot experiments. Logistic "
    "Regression and Linear SVM gave the strongest sparse-text rankings; the "
    "compact MLP was retained as the strongest practical non-linear candidate. "
    "No excluded algorithm is rerun in this assessed script and no web or "
    "leaderboard-solution research informed the selection."
)
DISPLAY_LABEL = {
    "toxic": "Toxic",
    "severe_toxic": "Severe toxicity",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity-based hate",
    "clean": "Clean",
}

TOKEN_RE = re.compile(r"(?u)\b[\w']+\b")
SENTENCE_BOUNDARY_RE = re.compile(r"(?:[.!?]+(?=\s|$)|\n+)")
WHITESPACE_RE = re.compile(r"\s+")


def _default_linear_parameters() -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Return representation- and label-specific settings for the linear models."""

    rare_labels = {"severe_toxic", "threat", "identity_hate"}
    character_logistic_c = {
        "toxic": 4.0,
        "severe_toxic": 2.0,
        "obscene": 4.0,
        "threat": 4.0,
        "insult": 2.0,
        "identity_hate": 2.0,
    }
    parameters: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for representation in REPRESENTATIONS:
        parameters[representation] = {
            "Logistic Regression": {},
            "Linear SVM": {},
        }
        for label in LABEL_COLUMNS:
            if representation == "Word TF-IDF":
                logistic_c = 4.0
                logistic_weight = None
                svm_c = 0.5
                svm_weight = "balanced" if label in rare_labels else None
            else:
                logistic_c = character_logistic_c[label]
                logistic_weight = None
                svm_c = 0.2
                svm_weight = None
            parameters[representation]["Logistic Regression"][label] = {
                "C": logistic_c,
                "class_weight": logistic_weight,
            }
            parameters[representation]["Linear SVM"][label] = {
                "C": svm_c,
                "class_weight": svm_weight,
            }
    return parameters


def _default_mlp_parameters() -> dict[str, dict[str, Any]]:
    """Return compact shared-output MLP settings for each representation."""

    return {
        "Word TF-IDF": {
            "hidden_layer_sizes": (64,),
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
        },
        "Character Bag-of-N-grams": {
            "hidden_layer_sizes": (64,),
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
        },
    }


@dataclass
class RunConfig:
    """Central, serialisable settings for one complete experiment."""

    output_dir: Path = Path("toxic_comment_three_model_outputs")
    seed: int = 42
    n_splits: int = 3
    threshold: float = 0.5
    meta_fraction: float = 0.16
    meta_audit_fraction: float = 0.50
    use_ensemble: bool = False
    ensemble_weight_mode: str = "global"
    ensemble_search_iterations: int = 384
    leaderboard_reference_auc: float = 0.98856
    observed_kaggle_score: float | None = None

    word_ngram_min: int = 1
    word_ngram_max: int = 2
    word_min_df: int = 2
    word_max_df: float = 0.995
    word_max_features: int = 250_000

    character_ngram_min: int = 3
    character_ngram_max: int = 5
    character_min_df: int = 2
    character_max_df: float = 0.995
    character_max_features: int = 300_000

    linear_max_iter: int = 5_000
    logistic_solver: str = "liblinear"

    mlp_svd_components: int = 192
    mlp_svd_iterations: int = 4
    mlp_batch_size: int = 1_024
    mlp_max_iter: int = 40
    mlp_balance_weight_cap: float = 8.0

    eda_top_words: int = 15
    eda_vocabulary_limit: int = 60_000
    row_limit: int | None = None
    run_eda: bool = True
    create_submissions: bool = True

    label_columns: tuple[str, ...] = field(default=LABEL_COLUMNS)
    text_column: str = TEXT_COLUMN
    id_column: str = ID_COLUMN
    linear_parameters: dict[str, dict[str, dict[str, dict[str, Any]]]] = field(
        default_factory=_default_linear_parameters
    )
    mlp_parameters: dict[str, dict[str, Any]] = field(
        default_factory=_default_mlp_parameters
    )

    def validate(self) -> None:
        """Reject settings that would weaken the intended experimental design."""

        if self.n_splits < 2:
            raise ValueError("The number of folds must be at least two.")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("The classification threshold must lie between zero and one.")
        if self.use_ensemble and not 0.05 <= self.meta_fraction < 0.40:
            raise ValueError("The meta-holdout fraction must lie between 0.05 and 0.40.")
        if not 0.20 <= self.meta_audit_fraction <= 0.80:
            raise ValueError("The audit share of the meta-holdout must lie between 0.20 and 0.80.")
        if self.ensemble_weight_mode not in {"global", "per-label"}:
            raise ValueError("Ensemble weight mode must be 'global' or 'per-label'.")
        if self.mlp_svd_components < 8:
            raise ValueError("The MLP SVD dimension must be at least eight.")
        if self.observed_kaggle_score is not None and not 0.0 <= self.observed_kaggle_score <= 1.0:
            raise ValueError("The observed Kaggle score must lie between zero and one.")
        if tuple(self.label_columns) != LABEL_COLUMNS:
            raise ValueError("The six fixed Kaggle label columns must remain unchanged.")
        _assert_experiment_grid()


@dataclass(frozen=True)
class DataPaths:
    """Resolved local paths for the labelled and optional competition data."""

    train: Path
    test: Path | None
    sample_submission: Path | None


@dataclass(frozen=True)
class GroupStructure:
    """Duplicate groups and their aggregated multilabel signatures."""

    row_group: np.ndarray
    group_keys: np.ndarray
    group_labels: np.ndarray
    group_sizes: np.ndarray


@dataclass(frozen=True)
class SplitPlan:
    """Row indices for development CV and the two meta-holdout partitions."""

    development_rows: np.ndarray
    meta_optimisation_rows: np.ndarray
    meta_audit_rows: np.ndarray
    cv_splits: tuple[tuple[np.ndarray, np.ndarray], ...]


@dataclass
class FittedCandidate:
    """A fitted candidate and its optional fold-local dimensionality adapter."""

    family: str
    representation: str
    estimators: Any
    svd: TruncatedSVD | None = None
    scaler: StandardScaler | None = None
    training_diagnostics: dict[str, Any] | None = None


def _assert_experiment_grid() -> None:
    """Enforce the exact three-by-two experimental grid."""

    expected_models = {
        "Logistic Regression",
        "Linear SVM",
        "Multilayer Perceptron",
    }
    expected_representations = {"Word TF-IDF", "Character Bag-of-N-grams"}
    if (
        len(MODEL_FAMILIES) != 3
        or len(set(MODEL_FAMILIES)) != 3
        or set(MODEL_FAMILIES) != expected_models
        or len(REPRESENTATIONS) != 2
        or len(set(REPRESENTATIONS)) != 2
        or set(REPRESENTATIONS) != expected_representations
        or len(candidate_keys()) != 6
    ):
        raise RuntimeError(
            "The assessment grid must contain exactly three fixed model families "
            "and two fixed feature-extraction methods."
        )


def candidate_key(representation: str, family: str) -> str:
    """Build a stable key for one representation–model combination."""

    return f"{REPRESENTATION_KEY[representation]}__{MODEL_KEY[family]}"


def candidate_keys() -> tuple[str, ...]:
    """Return the six candidate keys in a fixed order."""

    return tuple(
        candidate_key(representation, family)
        for representation in REPRESENTATIONS
        for family in MODEL_FAMILIES
    )


def candidate_parts(key: str) -> tuple[str, str]:
    """Recover the display names represented by a candidate key."""

    for representation in REPRESENTATIONS:
        for family in MODEL_FAMILIES:
            if candidate_key(representation, family) == key:
                return representation, family
    raise KeyError(f"Unknown candidate key: {key}")


def set_reproducible_seed(seed: int) -> None:
    """Set process-level pseudo-random seeds used by the pipeline."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _json_ready(value: Any) -> Any:
    """Convert nested configuration values into JSON-compatible objects."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a deterministic, human-readable JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)


def prepare_output_directory(path: Path) -> None:
    """Require a fresh output directory so stale results cannot survive a rerun."""

    if path.exists() and any(path.iterdir()):
        raise ValueError(
            f"The output directory is not empty: {path}. Choose a fresh directory "
            "to preserve an unambiguous experiment record."
        )
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of a local input without loading it at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_physical_type(path: Path) -> str:
    """Identify CSV, GZIP, ZIP and common erroneous downloads from their bytes."""

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {path}")
    with path.open("rb") as handle:
        prefix = handle.read(4096)
    lower = prefix.lstrip().lower()
    if zipfile.is_zipfile(path):
        return "zip"
    if prefix.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return "corrupt_zip"
    if prefix.startswith(b"\x1f\x8b"):
        return "gzip"
    if lower.startswith((b"<!doctype html", b"<html", b"<?xml")):
        return "html"
    if lower.startswith((b"{", b"[")):
        return "json_or_text"
    try:
        first_line = prefix.decode("utf-8-sig").splitlines()[0]
    except (UnicodeDecodeError, IndexError):
        return "unknown"
    return "csv" if "," in first_line else "plain_text"


def _read_zip_csv(path: Path, expected_stem: str | None) -> pd.DataFrame:
    """Read one CSV from a ZIP, including one nested official ZIP or GZIP member."""

    with zipfile.ZipFile(path, "r") as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if expected_stem:
            preferred = [
                name
                for name in members
                if Path(name).name.lower()
                in {
                    f"{expected_stem}.csv",
                    f"{expected_stem}.csv.gz",
                    f"{expected_stem}.csv.zip",
                    f"{expected_stem}.zip",
                }
            ]
        else:
            preferred = []
        candidates = preferred or [
            name
            for name in members
            if name.lower().endswith((".csv", ".csv.gz", ".csv.zip", ".zip"))
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"{path.name} must resolve to exactly one CSV for "
                f"'{expected_stem or 'the requested table'}'; candidates were {candidates}."
            )
        member = candidates[0]
        payload = archive.read(member)
        lower_name = member.lower()
        if lower_name.endswith(".csv"):
            return pd.read_csv(io.BytesIO(payload), low_memory=False)
        if lower_name.endswith(".gz"):
            with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as stream:
                return pd.read_csv(stream, low_memory=False)
        if lower_name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(payload), "r") as nested:
                csv_members = [
                    name
                    for name in nested.namelist()
                    if not name.endswith("/") and name.lower().endswith(".csv")
                ]
                if len(csv_members) != 1:
                    raise ValueError(
                        f"Nested archive {member} must contain exactly one CSV; "
                        f"found {csv_members}."
                    )
                with nested.open(csv_members[0], "r") as stream:
                    return pd.read_csv(stream, low_memory=False)
    raise RuntimeError(f"Could not read a CSV from {path}.")


def read_csv_robust(path: Path, expected_stem: str | None = None) -> pd.DataFrame:
    """Read a plain, GZIP or ZIP CSV by content rather than filename alone."""

    physical_type = detect_physical_type(path)
    LOGGER.info(
        "Reading %s (%.2f MiB; physical type: %s)",
        path,
        path.stat().st_size / (1024**2),
        physical_type,
    )
    if physical_type == "zip":
        return _read_zip_csv(path, expected_stem)
    if physical_type == "gzip":
        return pd.read_csv(path, compression="gzip", low_memory=False)
    if physical_type == "csv":
        if path.suffix.lower() in {".zip", ".gz"}:
            warnings.warn(
                f"{path.name} has a compressed-file suffix but contains plain CSV data.",
                RuntimeWarning,
            )
        return pd.read_csv(path, compression=None, low_memory=False)
    diagnostic = {
        "html": "an HTML page, possibly a saved sign-in page",
        "json_or_text": "a JSON or textual response rather than the dataset",
        "corrupt_zip": "an incomplete or corrupt ZIP archive",
        "plain_text": "plain text without a CSV header",
        "unknown": "an unrecognised binary format",
    }.get(physical_type, physical_type)
    raise ValueError(f"Cannot read {path.name}: its content is {diagnostic}.")


def _find_named_input(data_dir: Path, stem: str) -> Path | None:
    """Locate a named table recursively, favouring explicit filenames."""

    names = (
        f"{stem}.csv",
        f"{stem}.csv.gz",
        f"{stem}.csv.zip",
        f"{stem}.zip",
        f"{stem}.gz",
    )
    for name in names:
        direct = data_dir / name
        if direct.is_file():
            return direct
    if data_dir.exists():
        for name in names:
            matches = sorted(data_dir.rglob(name))
            if matches:
                return matches[0]
    outer_names = (
        "jigsaw-toxic-comment-classification-challenge.zip",
        "toxic-comment-classification-challenge.zip",
    )
    for name in outer_names:
        candidate = data_dir / name
        if candidate.is_file():
            return candidate
    return None


def resolve_data_paths(args: argparse.Namespace) -> DataPaths:
    """Resolve explicit input arguments or search the nominated data directory."""

    data_dir = Path(args.data_dir).expanduser().resolve()
    train = Path(args.train).expanduser().resolve() if args.train else _find_named_input(data_dir, "train")
    if args.benchmark_only:
        test = None
        sample = None
    else:
        test = Path(args.test).expanduser().resolve() if args.test else _find_named_input(data_dir, "test")
        sample = (
            Path(args.sample_submission).expanduser().resolve()
            if args.sample_submission
            else _find_named_input(data_dir, "sample_submission")
        )
    if train is None:
        raise FileNotFoundError(
            "No training table was found. Supply --train or place train.csv, "
            "train.csv.gz or train.csv.zip in --data-dir."
        )
    if (test is None) != (sample is None):
        raise ValueError(
            "Test data and a sample submission must be supplied together, or both omitted."
        )
    return DataPaths(train=train, test=test, sample_submission=sample)


def validate_training_schema(train: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    """Validate and clean the labelled training table without altering labels."""

    required = {cfg.id_column, cfg.text_column, *cfg.label_columns}
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"The training table is missing columns: {missing}")
    if train[cfg.id_column].isna().any():
        raise ValueError("The training identifier column contains missing values.")
    if not train[cfg.id_column].astype(str).is_unique:
        raise ValueError("Training identifiers must be unique.")
    labels = train.loc[:, list(cfg.label_columns)]
    if labels.isna().to_numpy().any() or not labels.isin([0, 1]).to_numpy().all():
        raise ValueError("All six training labels must be non-missing binary values.")
    cleaned = train.copy()
    missing_text = int(cleaned[cfg.text_column].isna().sum())
    if missing_text:
        LOGGER.warning("Replacing %d missing training comments with empty strings.", missing_text)
    cleaned[cfg.text_column] = cleaned[cfg.text_column].fillna("").astype(str)
    return cleaned


def validate_competition_schema(
    test: pd.DataFrame,
    sample: pd.DataFrame,
    cfg: RunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the optional test and sample-submission tables."""

    test_required = {cfg.id_column, cfg.text_column}
    sample_required = {cfg.id_column, *cfg.label_columns}
    missing_test = sorted(test_required.difference(test.columns))
    missing_sample = sorted(sample_required.difference(sample.columns))
    if missing_test or missing_sample:
        raise ValueError(
            f"Competition schema mismatch: test missing {missing_test}; "
            f"sample submission missing {missing_sample}."
        )
    if len(test) != len(sample):
        raise ValueError("Test and sample-submission row counts differ.")
    if test[cfg.id_column].isna().any() or sample[cfg.id_column].isna().any():
        raise ValueError("Test and sample-submission identifiers must not be missing.")
    test_ids = test[cfg.id_column].astype(str)
    sample_ids = sample[cfg.id_column].astype(str)
    if not test_ids.is_unique or not sample_ids.is_unique:
        raise ValueError("Test and sample-submission identifiers must be unique.")
    if set(test_ids) != set(sample_ids):
        raise ValueError("Test and sample-submission identifier sets differ.")
    cleaned_test = test.copy()
    cleaned_test[cfg.text_column] = cleaned_test[cfg.text_column].fillna("").astype(str)
    return cleaned_test, sample.copy()


def deterministic_row_limit(train: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    """Take a reproducible development subset while retaining every positive label."""

    if limit is None or limit >= len(train):
        return train.reset_index(drop=True)
    if limit < 1_000:
        raise ValueError("A row limit below 1,000 is too small for the six rare labels.")
    sampled = train.sample(n=limit, random_state=seed)
    additions: list[pd.DataFrame] = []
    for label in LABEL_COLUMNS:
        if int(sampled[label].sum()) == 0:
            additions.append(train.loc[train[label].eq(1)].sample(n=1, random_state=seed))
    if additions:
        sampled = pd.concat([sampled, *additions]).drop_duplicates(subset=[ID_COLUMN])
    LOGGER.warning(
        "Using a development row limit: %d of %d comments.", len(sampled), len(train)
    )
    return sampled.reset_index(drop=True)


def canonicalise_for_grouping(text: str) -> str:
    """Canonicalise only enough to detect case and spacing variants of duplicates."""

    normalised = unicodedata.normalize("NFKC", str(text)).casefold()
    return WHITESPACE_RE.sub(" ", normalised).strip()


def build_group_structure(texts: Sequence[str], y: np.ndarray) -> GroupStructure:
    """Create duplicate groups and aggregate their labels with logical OR."""

    canonical = pd.Series(texts, copy=False).map(canonicalise_for_grouping)
    row_group, unique_keys = pd.factorize(canonical, sort=False)
    n_groups = len(unique_keys)
    group_labels = np.zeros((n_groups, y.shape[1]), dtype=np.int8)
    np.maximum.at(group_labels, row_group, y)
    group_sizes = np.bincount(row_group, minlength=n_groups).astype(np.int64)
    return GroupStructure(
        row_group=row_group.astype(np.int64),
        group_keys=np.asarray(unique_keys, dtype=str),
        group_labels=group_labels,
        group_sizes=group_sizes,
    )


def _iterative_splitter_classes() -> tuple[Any, Any]:
    """Import iterative multilabel splitters only when an experiment is run."""

    try:
        from iterstrat.ml_stratifiers import (
            MultilabelStratifiedKFold,
            MultilabelStratifiedShuffleSplit,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The pinned dependency 'iterative-stratification==0.1.9' is required. "
            "Install the pinned dependencies in the active environment before "
            "running the experiment."
        ) from exc
    installed_version = importlib.metadata.version("iterative-stratification")
    if installed_version != "0.1.9":
        raise RuntimeError(
            "Reproducible splitting requires iterative-stratification==0.1.9; "
            f"the active version is {installed_version}."
        )
    return MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit


def _split_group_ids(
    group_ids: np.ndarray,
    group_labels: np.ndarray,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one iterative multilabel split over whole duplicate groups."""

    _, ShuffleSplitter = _iterative_splitter_classes()
    splitter = ShuffleSplitter(n_splits=1, test_size=test_fraction, random_state=seed)
    local_labels = group_labels[group_ids]
    dummy = np.zeros((len(group_ids), 1), dtype=np.int8)
    train_local, test_local = next(splitter.split(dummy, local_labels))
    return group_ids[train_local], group_ids[test_local]


def _rows_for_groups(row_group: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    """Map selected duplicate groups back to sorted row indices."""

    return np.flatnonzero(np.isin(row_group, group_ids)).astype(np.int64)


def build_split_plan(groups: GroupStructure, y: np.ndarray, cfg: RunConfig) -> SplitPlan:
    """Reserve a meta-holdout and create duplicate-safe iterative CV folds."""

    KFoldSplitter, _ = _iterative_splitter_classes()
    all_groups = np.arange(len(groups.group_keys), dtype=np.int64)
    if cfg.use_ensemble:
        development_groups, meta_groups = _split_group_ids(
            all_groups, groups.group_labels, cfg.meta_fraction, cfg.seed + 101
        )
        optimisation_groups, audit_groups = _split_group_ids(
            meta_groups,
            groups.group_labels,
            cfg.meta_audit_fraction,
            cfg.seed + 202,
        )
    else:
        development_groups = all_groups
        optimisation_groups = np.empty(0, dtype=np.int64)
        audit_groups = np.empty(0, dtype=np.int64)

    splitter = KFoldSplitter(
        n_splits=cfg.n_splits,
        shuffle=True,
        random_state=cfg.seed,
    )
    local_labels = groups.group_labels[development_groups]
    dummy = np.zeros((len(development_groups), 1), dtype=np.int8)
    cv_splits: list[tuple[np.ndarray, np.ndarray]] = []
    for train_local, valid_local in splitter.split(dummy, local_labels):
        train_groups = development_groups[train_local]
        valid_groups = development_groups[valid_local]
        train_rows = _rows_for_groups(groups.row_group, train_groups)
        valid_rows = _rows_for_groups(groups.row_group, valid_groups)
        if np.intersect1d(groups.row_group[train_rows], groups.row_group[valid_rows]).size:
            raise RuntimeError("A duplicate group crossed a CV boundary.")
        for label_index, label in enumerate(cfg.label_columns):
            if np.unique(y[train_rows, label_index]).size < 2:
                raise ValueError(
                    f"Training fold lacks both outcomes for '{label}'. "
                    "Use more data or fewer folds."
                )
            if np.unique(y[valid_rows, label_index]).size < 2:
                raise ValueError(
                    f"Validation fold lacks both outcomes for '{label}'. "
                    "Use more data or fewer folds."
                )
        cv_splits.append((train_rows, valid_rows))

    development_rows = _rows_for_groups(groups.row_group, development_groups)
    optimisation_rows = _rows_for_groups(groups.row_group, optimisation_groups)
    audit_rows = _rows_for_groups(groups.row_group, audit_groups)
    if cfg.use_ensemble:
        for partition_name, rows in (
            ("ensemble-optimisation", optimisation_rows),
            ("ensemble audit", audit_rows),
        ):
            invalid_labels = [
                label
                for label_index, label in enumerate(cfg.label_columns)
                if np.unique(y[rows, label_index]).size < 2
            ]
            if invalid_labels:
                raise ValueError(
                    f"The {partition_name} partition lacks both outcomes for "
                    f"{invalid_labels}. Increase --row-limit or omit --ensemble."
                )
    assigned = np.concatenate([development_rows, optimisation_rows, audit_rows])
    if len(np.unique(assigned)) != len(y) or len(assigned) != len(y):
        raise RuntimeError("Development and meta partitions do not cover each row exactly once.")
    if cfg.use_ensemble:
        partition_groups = [
            set(groups.row_group[development_rows]),
            set(groups.row_group[optimisation_rows]),
            set(groups.row_group[audit_rows]),
        ]
        if any(partition_groups[i] & partition_groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise RuntimeError("A duplicate group crossed a meta-holdout boundary.")

    LOGGER.info(
        "Split plan: %d development rows, %d ensemble-optimisation rows and %d audit rows.",
        len(development_rows),
        len(optimisation_rows),
        len(audit_rows),
    )
    return SplitPlan(
        development_rows=development_rows,
        meta_optimisation_rows=optimisation_rows,
        meta_audit_rows=audit_rows,
        cv_splits=tuple(cv_splits),
    )


def sentence_like_count(text: str) -> int:
    """Count reproducible sentence-like units without claiming linguistic parsing."""

    stripped = str(text).strip()
    if not stripped:
        return 0
    parts = [part for part in SENTENCE_BOUNDARY_RE.split(stripped) if part.strip()]
    return max(1, len(parts))


def run_eda(train: pd.DataFrame, cfg: RunConfig) -> dict[str, Any]:
    """Produce the class, comment, sentence-like, token and word analyses."""

    LOGGER.info("Running exploratory data analysis.")
    tables_dir = cfg.output_dir / "tables"
    figures_dir = cfg.output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    texts = train[cfg.text_column].astype(str)
    y = train.loc[:, list(cfg.label_columns)].to_numpy(dtype=np.int8)
    token_counts = texts.map(lambda text: len(TOKEN_RE.findall(text))).to_numpy(dtype=np.int64)
    sentence_counts = texts.map(sentence_like_count).to_numpy(dtype=np.int64)
    class_masks: dict[str, np.ndarray] = {
        label: y[:, index].astype(bool) for index, label in enumerate(cfg.label_columns)
    }
    class_masks["clean"] = y.sum(axis=1).eq(0) if isinstance(y, pd.DataFrame) else y.sum(axis=1) == 0

    rows: list[dict[str, Any]] = []
    for label, mask in class_masks.items():
        count = int(mask.sum())
        rows.append(
            {
                "class": label,
                "display_class": DISPLAY_LABEL[label],
                "comment_count": count,
                "comment_prevalence": count / len(train),
                "sentence_like_units": int(sentence_counts[mask].sum()),
                "tokens": int(token_counts[mask].sum()),
                "mean_sentence_like_units_per_comment": float(sentence_counts[mask].mean()),
                "mean_tokens_per_comment": float(token_counts[mask].mean()),
                "median_tokens_per_comment": float(np.median(token_counts[mask])),
            }
        )
    class_summary = pd.DataFrame(rows)
    class_summary.to_csv(tables_dir / "eda_class_sentence_token_summary.csv", index=False)

    # One shared unigram vocabulary makes the per-class raw-frequency comparison
    # efficient and directly comparable.  English stop words are removed only for
    # this descriptive table, never from the predictive representations.
    word_counter = CountVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        token_pattern=r"(?u)\b[\w']{2,}\b",
        ngram_range=(1, 1),
        min_df=2,
        max_features=cfg.eda_vocabulary_limit,
        dtype=np.int32,
    )
    word_matrix = word_counter.fit_transform(texts)
    terms = np.asarray(word_counter.get_feature_names_out())
    word_rows: list[dict[str, Any]] = []
    for label, mask in class_masks.items():
        totals = np.asarray(word_matrix[mask].sum(axis=0)).ravel()
        top_indices = np.argsort(totals)[::-1][: cfg.eda_top_words]
        for rank, index in enumerate(top_indices, start=1):
            if totals[index] <= 0:
                continue
            word_rows.append(
                {
                    "class": label,
                    "display_class": DISPLAY_LABEL[label],
                    "rank": rank,
                    "word": terms[index],
                    "frequency": int(totals[index]),
                }
            )
    common_words = pd.DataFrame(word_rows)
    common_words.to_csv(tables_dir / "eda_common_words_by_class.csv", index=False)

    figure_rows = class_summary[class_summary["class"] != "clean"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(
        figure_rows["display_class"],
        figure_rows["comment_prevalence"],
        color="#24557a",
    )
    ax.set_title("Training-label prevalence")
    ax.set_ylabel("Proportion of training comments")
    ax.set_xlabel("Toxicity label")
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, figure_rows["comment_prevalence"]):
        ax.annotate(
            f"{value:.2%}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(figures_dir / "eda_label_prevalence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(
        class_summary["display_class"],
        class_summary["mean_tokens_per_comment"],
        color="#4c8f76",
    )
    ax.set_title("Mean token count by overlapping class")
    ax.set_ylabel("Mean tokens per comment")
    ax.set_xlabel("Comment class")
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, class_summary["mean_tokens_per_comment"]):
        ax.annotate(
            f"{value:.1f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(figures_dir / "eda_mean_tokens_by_class.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "class_summary": class_summary,
        "common_words": common_words,
        "total_comments": len(train),
        "total_tokens": int(token_counts.sum()),
        "total_sentence_like_units": int(sentence_counts.sum()),
    }


def build_vectoriser(representation: str, cfg: RunConfig) -> Any:
    """Build one of the two permitted feature-extraction methods.

    The first method uses word-level TF-IDF. The alternative method uses binary
    character n-gram presence followed by L2 normalisation. This changes both
    the linguistic unit and the weighting scheme, rather than presenting two
    tokenisations as if they were two different extraction methods.
    """

    common: dict[str, Any] = {
        "lowercase": True,
        "strip_accents": "unicode",
        "sublinear_tf": True,
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "dtype": np.float32,
    }
    if representation == "Word TF-IDF":
        return TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b[\w']+\b",
            ngram_range=(cfg.word_ngram_min, cfg.word_ngram_max),
            min_df=cfg.word_min_df,
            max_df=cfg.word_max_df,
            max_features=cfg.word_max_features,
            **common,
        )
    if representation == "Character Bag-of-N-grams":
        return make_pipeline(
            CountVectorizer(
                lowercase=True,
                strip_accents="unicode",
                analyzer="char",
                ngram_range=(cfg.character_ngram_min, cfg.character_ngram_max),
                min_df=cfg.character_min_df,
                max_df=cfg.character_max_df,
                max_features=cfg.character_max_features,
                binary=True,
                dtype=np.float32,
            ),
            Normalizer(norm="l2", copy=False),
        )
    raise KeyError(f"Unknown representation: {representation}")


def matrix_mebibytes(matrix: Any) -> float:
    """Estimate the memory held by a dense or sparse feature matrix."""

    if hasattr(matrix, "data") and hasattr(matrix, "indices") and hasattr(matrix, "indptr"):
        return float(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes) / (1024**2)
    return float(np.asarray(matrix).nbytes) / (1024**2)


def _linear_estimator(
    family: str,
    representation: str,
    label: str,
    cfg: RunConfig,
    seed_offset: int,
) -> Any:
    """Create one label-specific linear estimator from the documented settings."""

    parameters = cfg.linear_parameters[representation][family][label]
    if family == "Logistic Regression":
        return LogisticRegression(
            C=float(parameters["C"]),
            class_weight=parameters.get("class_weight"),
            solver=cfg.logistic_solver,
            max_iter=cfg.linear_max_iter,
            random_state=cfg.seed + seed_offset,
        )
    if family == "Linear SVM":
        return LinearSVC(
            C=float(parameters["C"]),
            class_weight=parameters.get("class_weight"),
            dual=True,
            max_iter=cfg.linear_max_iter,
            random_state=cfg.seed + seed_offset,
        )
    raise KeyError(f"Not a linear family: {family}")


def _fit_mlp_adapter(
    X_train: Any,
    representation: str,
    cfg: RunConfig,
    seed_offset: int,
) -> tuple[np.ndarray, TruncatedSVD, StandardScaler]:
    """Fit the MLP's compact SVD representation on training rows only."""

    maximum = min(X_train.shape[0] - 1, X_train.shape[1] - 1)
    components = min(cfg.mlp_svd_components, maximum)
    if components < 8:
        raise ValueError("The fold is too small for the configured MLP SVD adapter.")
    svd = TruncatedSVD(
        n_components=components,
        n_iter=cfg.mlp_svd_iterations,
        random_state=cfg.seed + seed_offset,
    )
    reduced = svd.fit_transform(X_train).astype(np.float32, copy=False)
    scaler = StandardScaler(copy=False)
    scaled = scaler.fit_transform(reduced).astype(np.float32, copy=False)
    LOGGER.info(
        "MLP adapter for %s retained %d components and %.2f%% explained variance.",
        representation,
        components,
        100.0 * float(np.sum(svd.explained_variance_ratio_)),
    )
    return scaled, svd, scaler


def _transform_mlp_features(
    X: Any,
    svd: TruncatedSVD,
    scaler: StandardScaler,
) -> np.ndarray:
    """Apply a training-fitted MLP adapter to unseen feature rows."""

    reduced = svd.transform(X).astype(np.float32, copy=False)
    return scaler.transform(reduced).astype(np.float32, copy=False)


def fit_candidate(
    family: str,
    representation: str,
    X_train: Any,
    y_train: np.ndarray,
    cfg: RunConfig,
    seed_offset: int,
) -> FittedCandidate:
    """Fit one of the three permitted model families."""

    if family in {"Logistic Regression", "Linear SVM"}:
        estimators: list[Any] = []
        for label_index, label in enumerate(cfg.label_columns):
            if np.unique(y_train[:, label_index]).size < 2:
                raise ValueError(f"Cannot fit '{label}': the training partition has one outcome.")
            estimator = _linear_estimator(
                family,
                representation,
                label,
                cfg,
                seed_offset + label_index,
            )
            estimator.fit(X_train, y_train[:, label_index])
            estimators.append(estimator)
        return FittedCandidate(
            family=family,
            representation=representation,
            estimators=estimators,
        )

    if family == "Multilayer Perceptron":
        X_train_mlp, svd, scaler = _fit_mlp_adapter(
            X_train, representation, cfg, seed_offset
        )
        parameters = cfg.mlp_parameters[representation]
        estimator = MLPClassifier(
            hidden_layer_sizes=tuple(parameters["hidden_layer_sizes"]),
            activation="relu",
            solver="adam",
            alpha=float(parameters["alpha"]),
            batch_size=cfg.mlp_batch_size,
            learning_rate="constant",
            learning_rate_init=float(parameters["learning_rate_init"]),
            max_iter=cfg.mlp_max_iter,
            early_stopping=False,
            random_state=cfg.seed + seed_offset,
        )
        # A single shared network retains the efficient native six-output design.
        # Row weights average the inverse-frequency binary weights over the six
        # labels, so rare outcomes influence the shared loss without six separate
        # networks.  The cap prevents multiply-labelled comments from dominating.
        positive_counts = y_train.sum(axis=0).astype(np.float64)
        negative_counts = len(y_train) - positive_counts
        if np.any(positive_counts == 0) or np.any(negative_counts == 0):
            raise ValueError("The MLP training partition must contain both outcomes for every label.")
        positive_weights = len(y_train) / (2.0 * positive_counts)
        negative_weights = len(y_train) / (2.0 * negative_counts)
        binary_weights = np.where(y_train == 1, positive_weights, negative_weights)
        sample_weights = binary_weights.mean(axis=1)
        sample_weights /= sample_weights.mean()
        sample_weights = np.clip(sample_weights, 0.0, cfg.mlp_balance_weight_cap)
        sample_weights /= sample_weights.mean()
        if "sample_weight" not in inspect.signature(estimator.fit).parameters:
            raise RuntimeError(
                "The balanced MLP requires scikit-learn 1.7 or later. Install the "
                "pinned dependencies before running the experiment."
            )
        LOGGER.info(
            "Training the %s MLP for a maximum of %d fixed epochs.",
            representation,
            cfg.mlp_max_iter,
        )
        # Reaching the fixed epoch budget is expected and does not invalidate
        # ROC-AUC evaluation. Replace scikit-learn's repeated warning with one
        # structured record containing the completed epochs and final loss.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            estimator.fit(X_train_mlp, y_train, sample_weight=sample_weights)
        diagnostics = {
            "mlp_epochs_completed": int(estimator.n_iter_),
            "mlp_final_loss": float(estimator.loss_),
            "mlp_reached_epoch_budget": bool(
                estimator.n_iter_ >= cfg.mlp_max_iter
            ),
        }
        LOGGER.info(
            "The %s MLP completed %d epochs; final training loss %.6f; "
            "epoch budget reached: %s.",
            representation,
            diagnostics["mlp_epochs_completed"],
            diagnostics["mlp_final_loss"],
            diagnostics["mlp_reached_epoch_budget"],
        )
        return FittedCandidate(
            family=family,
            representation=representation,
            estimators=estimator,
            svd=svd,
            scaler=scaler,
            training_diagnostics=diagnostics,
        )

    raise KeyError(f"Unknown model family: {family}")


def predict_candidate(model: FittedCandidate, X: Any) -> np.ndarray:
    """Return six continuous scores in [0, 1] for a fitted candidate."""

    if model.family == "Logistic Regression":
        columns = [estimator.predict_proba(X)[:, 1] for estimator in model.estimators]
        scores = np.column_stack(columns)
    elif model.family == "Linear SVM":
        # This bounded, strictly increasing transform preserves finite margin
        # ordering while retaining 0.5 as the SVM decision boundary. The values
        # are probability-like scores, not calibrated probabilities.
        columns = [
            0.5 + np.arctan(estimator.decision_function(X)) / np.pi
            for estimator in model.estimators
        ]
        scores = np.column_stack(columns)
    elif model.family == "Multilayer Perceptron":
        if model.svd is None or model.scaler is None:
            raise RuntimeError("The fitted MLP is missing its fold-local adapter.")
        X_mlp = _transform_mlp_features(X, model.svd, model.scaler)
        raw = model.estimators.predict_proba(X_mlp)
        if isinstance(raw, list):
            columns = [
                values[:, 1] if np.asarray(values).ndim == 2 else np.asarray(values)
                for values in raw
            ]
            scores = np.column_stack(columns)
        else:
            scores = np.asarray(raw)
    else:
        raise KeyError(f"Unknown model family: {model.family}")
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    if scores.shape[1] != len(LABEL_COLUMNS):
        raise RuntimeError(
            f"{model.family} returned {scores.shape[1]} columns; expected six."
        )
    return np.clip(scores, 1e-15, 1.0 - 1e-15).astype(np.float64, copy=False)


def calculate_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    labels: Sequence[str],
    threshold: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Calculate required aggregate and per-label classification metrics."""

    if y_true.shape != scores.shape:
        raise ValueError(f"Metric shapes differ: truth {y_true.shape}; scores {scores.shape}.")
    predictions = (scores >= threshold).astype(np.int8)
    aggregate: dict[str, float] = {
        # Element-wise accuracy is the proportion of correct label decisions.
        # Subset accuracy below is stricter: all six decisions must be correct.
        "element_wise_accuracy": float(np.mean(y_true == predictions)),
        "subset_accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "micro_precision": float(
            precision_score(y_true, predictions, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_true, predictions, average="micro", zero_division=0)
        ),
        "micro_f1": float(f1_score(y_true, predictions, average="micro", zero_division=0)),
    }
    per_label: list[dict[str, Any]] = []
    auc_values: list[float] = []
    for label_index, label in enumerate(labels):
        if np.unique(y_true[:, label_index]).size < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y_true[:, label_index], scores[:, label_index]))
        auc_values.append(auc)
        per_label.append(
            {
                "label": label,
                "accuracy": float(
                    accuracy_score(y_true[:, label_index], predictions[:, label_index])
                ),
                "precision": float(
                    precision_score(
                        y_true[:, label_index], predictions[:, label_index], zero_division=0
                    )
                ),
                "recall": float(
                    recall_score(
                        y_true[:, label_index], predictions[:, label_index], zero_division=0
                    )
                ),
                "f1": float(
                    f1_score(y_true[:, label_index], predictions[:, label_index], zero_division=0)
                ),
                "roc_auc": auc,
            }
        )
    aggregate["mean_column_roc_auc"] = float(np.nanmean(auc_values))
    return aggregate, per_label


def calculate_rank_auc(
    y_true: np.ndarray,
    rank_scores: np.ndarray,
    labels: Sequence[str],
) -> tuple[float, list[dict[str, Any]]]:
    """Calculate only rank-valid ROC-AUC metrics for an ensemble.

    Fractional ranks have no portable probability threshold: a value of 0.5
    simply marks the median prediction in the current batch. Accuracy,
    precision, recall and F1 are therefore deliberately omitted for the rank
    ensemble and remain available for all six individual candidates.
    """

    if y_true.shape != rank_scores.shape:
        raise ValueError(
            f"Rank-AUC shapes differ: truth {y_true.shape}; scores {rank_scores.shape}."
        )
    rows: list[dict[str, Any]] = []
    auc_values: list[float] = []
    for label_index, label in enumerate(labels):
        if np.unique(y_true[:, label_index]).size < 2:
            auc = float("nan")
        else:
            auc = float(
                roc_auc_score(y_true[:, label_index], rank_scores[:, label_index])
            )
        auc_values.append(auc)
        rows.append({"label": label, "roc_auc": auc})
    return float(np.nanmean(auc_values)), rows


def export_hyperparameters(cfg: RunConfig) -> None:
    """Export a report-facing record of label-specific and shared parameters."""

    rows: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        for family in ("Logistic Regression", "Linear SVM"):
            for label in LABEL_COLUMNS:
                values = cfg.linear_parameters[representation][family][label]
                rows.append(
                    {
                        "representation": representation,
                        "model_family": family,
                        "label": label,
                        "parameter_scope": "label-specific",
                        "parameters": json.dumps(_json_ready(values), sort_keys=True),
                    }
                )
        mlp_values = {
            **cfg.mlp_parameters[representation],
            "svd_components": cfg.mlp_svd_components,
            "batch_size": cfg.mlp_batch_size,
            "max_iter": cfg.mlp_max_iter,
            "early_stopping": False,
            "balance_weight_cap": cfg.mlp_balance_weight_cap,
            "balancing": "mean inverse-frequency binary row weight",
        }
        for label in LABEL_COLUMNS:
            rows.append(
                {
                    "representation": representation,
                    "model_family": "Multilayer Perceptron",
                    "label": label,
                    "parameter_scope": "shared native multilabel network with fixed epochs",
                    "parameters": json.dumps(_json_ready(mlp_values), sort_keys=True),
                }
            )
    pd.DataFrame(rows).to_csv(
        cfg.output_dir / "tables" / "model_hyperparameters.csv", index=False
    )


def run_cross_validation(
    train: pd.DataFrame,
    y: np.ndarray,
    plan: SplitPlan,
    cfg: RunConfig,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Run the exact three-by-two benchmark and retain development OOF scores."""

    _assert_experiment_grid()
    keys = candidate_keys()
    oof_scores = {
        key: np.full((len(train), len(cfg.label_columns)), np.nan, dtype=np.float64)
        for key in keys
    }
    fold_rows: list[dict[str, Any]] = []
    texts = train[cfg.text_column].astype(str).reset_index(drop=True)

    for fold_number, (train_rows, valid_rows) in enumerate(plan.cv_splits, start=1):
        LOGGER.info(
            "Starting fold %d of %d: %d training rows and %d validation rows.",
            fold_number,
            len(plan.cv_splits),
            len(train_rows),
            len(valid_rows),
        )
        for representation in REPRESENTATIONS:
            feature_start = time.perf_counter()
            vectoriser = build_vectoriser(representation, cfg)
            X_train = vectoriser.fit_transform(texts.iloc[train_rows])
            X_valid = vectoriser.transform(texts.iloc[valid_rows])
            feature_seconds = time.perf_counter() - feature_start
            LOGGER.info(
                "%s fold features: %s training shape, %.1f MiB, %.1f seconds.",
                representation,
                X_train.shape,
                matrix_mebibytes(X_train),
                feature_seconds,
            )
            for family_index, family in enumerate(MODEL_FAMILIES):
                key = candidate_key(representation, family)
                model_start = time.perf_counter()
                model = fit_candidate(
                    family,
                    representation,
                    X_train,
                    y[train_rows],
                    cfg,
                    seed_offset=fold_number * 100 + family_index * 10,
                )
                scores = predict_candidate(model, X_valid)
                model_seconds = time.perf_counter() - model_start
                if model.svd is not None:
                    adapter_name = (
                        f"Fold-local TruncatedSVD-{model.svd.n_components} + StandardScaler"
                    )
                    model_input_features = int(model.svd.n_components)
                    explained_variance = float(np.sum(model.svd.explained_variance_ratio_))
                else:
                    adapter_name = "None"
                    model_input_features = int(X_train.shape[1])
                    explained_variance = float("nan")
                training_diagnostics = model.training_diagnostics or {}
                oof_scores[key][valid_rows] = scores
                metrics, _ = calculate_metrics(
                    y[valid_rows], scores, cfg.label_columns, cfg.threshold
                )
                fold_rows.append(
                    {
                        "fold": fold_number,
                        "candidate": key,
                        "representation": representation,
                        "model_family": family,
                        "training_rows": len(train_rows),
                        "validation_rows": len(valid_rows),
                        "feature_count": X_train.shape[1],
                        "model_input_features": model_input_features,
                        "model_input_adapter": adapter_name,
                        "adapter_explained_variance_ratio": explained_variance,
                        "mlp_epochs_completed": training_diagnostics.get(
                            "mlp_epochs_completed"
                        ),
                        "mlp_final_loss": training_diagnostics.get("mlp_final_loss"),
                        "mlp_reached_epoch_budget": training_diagnostics.get(
                            "mlp_reached_epoch_budget"
                        ),
                        "feature_seconds": feature_seconds,
                        "model_seconds": model_seconds,
                        **metrics,
                    }
                )
                LOGGER.info(
                    "%s + %s: mean column-wise ROC-AUC %.6f; macro F1 %.6f; %.1f seconds.",
                    representation,
                    family,
                    metrics["mean_column_roc_auc"],
                    metrics["macro_f1"],
                    model_seconds,
                )
                del model, scores
                gc.collect()
            del vectoriser, X_train, X_valid
            gc.collect()

    development_rows = plan.development_rows
    coverage = np.zeros(len(train), dtype=np.int16)
    for _, valid_rows in plan.cv_splits:
        coverage[valid_rows] += 1
    if not np.all(coverage[development_rows] == 1):
        raise RuntimeError("Each development row must receive exactly one OOF prediction.")
    for key, values in oof_scores.items():
        if np.isnan(values[development_rows]).any():
            raise RuntimeError(f"OOF predictions are incomplete for {key}.")
        outside = np.setdiff1d(np.arange(len(train)), development_rows)
        if outside.size and not np.isnan(values[outside]).all():
            raise RuntimeError(f"OOF predictions for {key} leaked outside development rows.")

    summary_rows: list[dict[str, Any]] = []
    per_label_rows: list[dict[str, Any]] = []
    fold_results = pd.DataFrame(fold_rows)
    for key in keys:
        representation, family = candidate_parts(key)
        metrics, label_metrics = calculate_metrics(
            y[development_rows],
            oof_scores[key][development_rows],
            cfg.label_columns,
            cfg.threshold,
        )
        family_folds = fold_results.loc[fold_results["candidate"].eq(key)]
        summary_rows.append(
            {
                "candidate": key,
                "representation": representation,
                "model_family": family,
                "model_input_adapter": (
                    f"Fold-local TruncatedSVD-{cfg.mlp_svd_components} + StandardScaler"
                    if family == "Multilayer Perceptron"
                    else "None"
                ),
                **{f"oof_{name}": value for name, value in metrics.items()},
                "fold_roc_auc_mean": float(family_folds["mean_column_roc_auc"].mean()),
                "fold_roc_auc_standard_deviation": float(
                    family_folds["mean_column_roc_auc"].std(ddof=1)
                ),
                "total_model_seconds": float(family_folds["model_seconds"].sum()),
            }
        )
        for values in label_metrics:
            per_label_rows.append(
                {
                    "candidate": key,
                    "representation": representation,
                    "model_family": family,
                    **values,
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["fold_roc_auc_mean", "oof_mean_column_roc_auc"], ascending=False
    ).reset_index(drop=True)
    per_label_results = pd.DataFrame(per_label_rows)
    best_key = str(summary.iloc[0]["candidate"])

    tables_dir = cfg.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(tables_dir / "benchmark_fold_results.csv", index=False)
    summary.to_csv(tables_dir / "benchmark_summary.csv", index=False)
    per_label_results.to_csv(tables_dir / "benchmark_per_label_metrics.csv", index=False)

    cube = np.stack([oof_scores[key][development_rows] for key in keys], axis=1)
    np.savez_compressed(
        cfg.output_dir / "oof_predictions.npz",
        row_index=development_rows,
        identifiers=train.iloc[development_rows][cfg.id_column].astype(str).to_numpy(dtype=str),
        truth=y[development_rows],
        scores=cube,
        candidates=np.asarray(keys, dtype=str),
        labels=np.asarray(cfg.label_columns, dtype=str),
    )
    save_json(
        cfg.output_dir / "oof_predictions_manifest.json",
        {
            "format": "NumPy compressed archive",
            "score_shape": list(cube.shape),
            "axis_order": ["development row", "candidate", "label"],
            "candidates": list(keys),
            "labels": list(cfg.label_columns),
            "duplicate_safe": True,
            "vectorisers_fitted_within_each_fold": True,
        },
    )
    plot_benchmark(summary, per_label_results, cfg)
    LOGGER.info("Best single candidate from mean fold ROC-AUC: %s.", best_key)
    return oof_scores, fold_results, summary, per_label_results, best_key


def plot_benchmark(
    summary: pd.DataFrame,
    per_label: pd.DataFrame,
    cfg: RunConfig,
) -> None:
    """Save an aggregate comparison and a per-label ROC-AUC heat map."""

    figures_dir = cfg.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    ordered = summary.sort_values("fold_roc_auc_mean", ascending=True)
    labels = [
        f"{representation}\n{family}"
        for representation, family in zip(ordered["representation"], ordered["model_family"])
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, ordered["fold_roc_auc_mean"], color="#24557a")
    ax.set_title("Three models across two text representations")
    ax.set_xlabel("Mean fold column-wise ROC-AUC")
    lower = max(0.0, float(ordered["fold_roc_auc_mean"].min()) - 0.04)
    upper = min(1.0, float(ordered["fold_roc_auc_mean"].max()) + 0.015)
    ax.set_xlim(lower, upper)
    for bar, value in zip(bars, ordered["fold_roc_auc_mean"]):
        ax.annotate(
            f"{value:.5f}",
            (value, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha="left",
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(figures_dir / "benchmark_mean_roc_auc.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pivot = per_label.pivot(index="candidate", columns="label", values="roc_auc").loc[list(candidate_keys())]
    matrix = pivot.loc[:, list(cfg.label_columns)].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(matrix, aspect="auto", vmin=max(0.5, np.nanmin(matrix) - 0.02), vmax=1.0)
    ax.set_yticks(range(len(candidate_keys())))
    ax.set_yticklabels(
        [" + ".join(candidate_parts(key)) for key in candidate_keys()], fontsize=8
    )
    ax.set_xticks(range(len(cfg.label_columns)))
    ax.set_xticklabels([DISPLAY_LABEL[label] for label in cfg.label_columns], rotation=25, ha="right")
    ax.set_title("Development OOF ROC-AUC by label")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="ROC-AUC")
    fig.tight_layout()
    fig.savefig(figures_dir / "benchmark_per_label_roc_auc.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def fit_all_candidates_for_evaluation(
    train: pd.DataFrame,
    y: np.ndarray,
    training_rows: np.ndarray,
    evaluation_sets: Mapping[str, np.ndarray],
    cfg: RunConfig,
) -> dict[str, dict[str, np.ndarray]]:
    """Fit all six candidates once and score each named unseen partition."""

    texts = train[cfg.text_column].astype(str).reset_index(drop=True)
    outputs: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in evaluation_sets
    }
    for representation in REPRESENTATIONS:
        LOGGER.info("Fitting %s on the development partition for meta evaluation.", representation)
        vectoriser = build_vectoriser(representation, cfg)
        X_train = vectoriser.fit_transform(texts.iloc[training_rows])
        X_evaluations = {
            name: vectoriser.transform(texts.iloc[rows])
            for name, rows in evaluation_sets.items()
        }
        for family_index, family in enumerate(MODEL_FAMILIES):
            key = candidate_key(representation, family)
            model = fit_candidate(
                family,
                representation,
                X_train,
                y[training_rows],
                cfg,
                seed_offset=700 + family_index * 10,
            )
            for name, matrix in X_evaluations.items():
                outputs[name][key] = predict_candidate(model, matrix)
            del model
            gc.collect()
        del vectoriser, X_train, X_evaluations
        gc.collect()
    for name in outputs:
        if set(outputs[name]) != set(candidate_keys()):
            raise RuntimeError(f"Meta predictions are incomplete for {name}.")
    return outputs


def rank_prediction_cube(cube: np.ndarray) -> np.ndarray:
    """Convert each candidate and label column to fractional average ranks."""

    ranked = np.empty_like(cube, dtype=np.float64)
    n_rows = cube.shape[0]
    for candidate_index in range(cube.shape[1]):
        for label_index in range(cube.shape[2]):
            ranked[:, candidate_index, label_index] = rankdata(
                cube[:, candidate_index, label_index], method="average"
            ) / (n_rows + 1.0)
    return ranked


def _weight_candidates(n_candidates: int, iterations: int, seed: int) -> np.ndarray:
    """Create deterministic simplex candidates for rank-weight optimisation."""

    pool: list[np.ndarray] = [np.full(n_candidates, 1.0 / n_candidates)]
    pool.extend(np.eye(n_candidates))
    for first in range(n_candidates):
        for second in range(first + 1, n_candidates):
            for first_share in (0.25, 0.50, 0.75):
                weights = np.zeros(n_candidates)
                weights[first] = first_share
                weights[second] = 1.0 - first_share
                pool.append(weights)
    generator = np.random.default_rng(seed)
    if iterations:
        pool.extend(generator.dirichlet(np.full(n_candidates, 0.70), size=iterations))
    return np.asarray(pool, dtype=np.float64)


def optimise_rank_weights(
    truth: np.ndarray,
    score_cube: np.ndarray,
    cfg: RunConfig,
) -> tuple[np.ndarray, float]:
    """Learn non-negative rank weights on the optimisation half of the meta-holdout."""

    ranks = rank_prediction_cube(score_cube)
    pool = _weight_candidates(score_cube.shape[1], cfg.ensemble_search_iterations, cfg.seed + 303)
    if cfg.ensemble_weight_mode == "global":
        best_score = -math.inf
        best_weights = pool[0]
        for weights in pool:
            blended = np.einsum("ncl,c->nl", ranks, weights, optimize=True)
            aucs = [
                roc_auc_score(truth[:, label], blended[:, label])
                for label in range(truth.shape[1])
            ]
            score = float(np.mean(aucs))
            if score > best_score:
                best_score = score
                best_weights = weights.copy()
        return best_weights, best_score

    label_weights = np.empty((truth.shape[1], score_cube.shape[1]), dtype=np.float64)
    label_scores: list[float] = []
    for label in range(truth.shape[1]):
        best_score = -math.inf
        best_weights = pool[0]
        for weights in pool:
            blended = ranks[:, :, label] @ weights
            score = float(roc_auc_score(truth[:, label], blended))
            if score > best_score:
                best_score = score
                best_weights = weights.copy()
        label_weights[label] = best_weights
        label_scores.append(best_score)
    return label_weights, float(np.mean(label_scores))


def apply_rank_weights(score_cube: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply locked weights to one complete evaluation-batch prediction cube.

    Ranks are recomputed within the supplied batch. This transductive operation
    is suitable for a fixed Kaggle test set scored by ROC-AUC, but it is not a
    pointwise probability model for deployment.
    """

    ranks = rank_prediction_cube(score_cube)
    if weights.ndim == 1:
        blended = np.einsum("ncl,c->nl", ranks, weights, optimize=True)
    elif weights.ndim == 2:
        blended = np.einsum("ncl,lc->nl", ranks, weights, optimize=True)
    else:
        raise ValueError(f"Unexpected ensemble-weight shape: {weights.shape}")
    return np.clip(blended, 1e-15, 1.0 - 1e-15).astype(np.float64, copy=False)


def run_meta_ensemble(
    train: pd.DataFrame,
    y: np.ndarray,
    plan: SplitPlan,
    best_key: str,
    cfg: RunConfig,
) -> tuple[np.ndarray, bool, float, float]:
    """Optimise weights, audit them once and decide whether to use the blend."""

    LOGGER.info("Fitting the six locked candidates for leakage-aware meta blending.")
    predictions = fit_all_candidates_for_evaluation(
        train,
        y,
        plan.development_rows,
        {
            "optimisation": plan.meta_optimisation_rows,
            "audit": plan.meta_audit_rows,
        },
        cfg,
    )
    keys = candidate_keys()
    optimisation_cube = np.stack(
        [predictions["optimisation"][key] for key in keys], axis=1
    )
    audit_cube = np.stack([predictions["audit"][key] for key in keys], axis=1)
    weights, optimisation_auc = optimise_rank_weights(
        y[plan.meta_optimisation_rows], optimisation_cube, cfg
    )
    audit_blend = apply_rank_weights(audit_cube, weights)
    audit_auc, audit_per_label = calculate_rank_auc(
        y[plan.meta_audit_rows], audit_blend, cfg.label_columns
    )
    best_index = keys.index(best_key)
    best_audit_metrics, _ = calculate_metrics(
        y[plan.meta_audit_rows],
        audit_cube[:, best_index, :],
        cfg.label_columns,
        cfg.threshold,
    )
    LOGGER.info(
        "Rank ensemble: optimisation AUC %.6f; untouched audit AUC %.6f; "
        "best-single audit AUC %.6f.",
        optimisation_auc,
        audit_auc,
        best_audit_metrics["mean_column_roc_auc"],
    )
    accepted = audit_auc > best_audit_metrics["mean_column_roc_auc"]
    if not accepted:
        LOGGER.warning(
            "The rank ensemble did not beat the best single candidate on the "
            "untouched audit partition; no ensemble submission will be written."
        )

    weight_rows: list[dict[str, Any]] = []
    if weights.ndim == 1:
        for key, weight in zip(keys, weights):
            weight_rows.append(
                {"label": "All labels", "candidate": key, "weight": float(weight)}
            )
    else:
        for label_index, label in enumerate(cfg.label_columns):
            for candidate_index, key in enumerate(keys):
                weight_rows.append(
                    {
                        "label": label,
                        "candidate": key,
                        "weight": float(weights[label_index, candidate_index]),
                    }
                )
    pd.DataFrame(weight_rows).to_csv(
        cfg.output_dir / "tables" / "ensemble_weights.csv", index=False
    )
    result_rows = [
        {
            "partition": "Ensemble optimisation",
            "candidate": "Rank ensemble",
            "mean_column_roc_auc": optimisation_auc,
        },
        {
            "partition": "Untouched audit",
            "candidate": "Rank ensemble",
            "metric_scope": "ROC-AUC only; rank scores have no fixed class threshold",
            "mean_column_roc_auc": audit_auc,
            "accepted_for_submission": accepted,
        },
        {
            "partition": "Untouched audit",
            "candidate": best_key,
            "metric_scope": "All required thresholded metrics and ROC-AUC",
            "accepted_for_submission": not accepted,
            **best_audit_metrics,
        },
    ]
    pd.DataFrame(result_rows).to_csv(
        cfg.output_dir / "tables" / "ensemble_meta_results.csv", index=False
    )
    audit_label_table = pd.DataFrame(audit_per_label)
    audit_label_table.insert(0, "candidate", "Rank ensemble")
    audit_label_table.to_csv(
        cfg.output_dir / "tables" / "ensemble_audit_per_label_metrics.csv", index=False
    )
    np.savez_compressed(
        cfg.output_dir / "meta_predictions.npz",
        optimisation_row_index=plan.meta_optimisation_rows,
        audit_row_index=plan.meta_audit_rows,
        optimisation_truth=y[plan.meta_optimisation_rows],
        audit_truth=y[plan.meta_audit_rows],
        optimisation_scores=optimisation_cube,
        audit_scores=audit_cube,
        candidates=np.asarray(keys, dtype=str),
        labels=np.asarray(cfg.label_columns, dtype=str),
        weights=weights,
    )
    if weights.ndim == 1 and not np.isclose(weights.sum(), 1.0):
        raise RuntimeError("Global ensemble weights do not sum to one.")
    if weights.ndim == 2 and not np.allclose(weights.sum(axis=1), 1.0):
        raise RuntimeError("Per-label ensemble weights do not sum to one.")
    return (
        weights,
        accepted,
        audit_auc,
        float(best_audit_metrics["mean_column_roc_auc"]),
    )


def fit_full_candidate_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    required_keys: set[str],
    cfg: RunConfig,
) -> dict[str, np.ndarray]:
    """Refit requested candidates on all labelled rows and score the test data."""

    train_text = train[cfg.text_column].astype(str)
    test_text = test[cfg.text_column].astype(str)
    predictions: dict[str, np.ndarray] = {}
    for representation in REPRESENTATIONS:
        representation_keys = {
            candidate_key(representation, family) for family in MODEL_FAMILIES
        }
        if not required_keys.intersection(representation_keys):
            continue
        LOGGER.info("Full refit of %s.", representation)
        vectoriser = build_vectoriser(representation, cfg)
        X_train = vectoriser.fit_transform(train_text)
        X_test = vectoriser.transform(test_text)
        for family_index, family in enumerate(MODEL_FAMILIES):
            key = candidate_key(representation, family)
            if key not in required_keys:
                continue
            LOGGER.info("Full refit of %s + %s.", representation, family)
            model = fit_candidate(
                family,
                representation,
                X_train,
                y,
                cfg,
                seed_offset=900 + family_index * 10,
            )
            predictions[key] = predict_candidate(model, X_test)
            del model
            gc.collect()
        del vectoriser, X_train, X_test
        gc.collect()
    if set(predictions) != required_keys:
        raise RuntimeError(
            f"Full-refit predictions differ from those requested: {sorted(predictions)}."
        )
    return predictions


def make_submission(
    sample: pd.DataFrame,
    test: pd.DataFrame,
    scores: np.ndarray,
    output_path: Path,
    cfg: RunConfig,
) -> pd.DataFrame:
    """Align scores to the official identifier order and validate the CSV schema."""

    if scores.shape != (len(test), len(cfg.label_columns)):
        raise ValueError(
            f"Submission-score shape {scores.shape} does not match "
            f"({len(test)}, {len(cfg.label_columns)})."
        )
    score_table = pd.DataFrame(scores, columns=cfg.label_columns)
    score_table.index = test[cfg.id_column].astype(str)
    submission = sample[[cfg.id_column]].copy()
    sample_keys = submission[cfg.id_column].astype(str)
    for label in cfg.label_columns:
        submission[label] = sample_keys.map(score_table[label])
    submission = submission[[cfg.id_column, *cfg.label_columns]]
    if submission.loc[:, list(cfg.label_columns)].isna().to_numpy().any():
        raise RuntimeError("At least one sample-submission identifier lacks a prediction.")
    values = submission.loc[:, list(cfg.label_columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
        raise RuntimeError("Submission scores must be finite and lie in [0, 1].")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    LOGGER.info("Wrote %s with %d rows.", output_path, len(submission))
    return submission


def create_final_submissions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample: pd.DataFrame,
    y: np.ndarray,
    best_key: str,
    ensemble_weights: np.ndarray | None,
    cfg: RunConfig,
) -> dict[str, Path]:
    """Create distinct best-single and optional rank-ensemble submission files."""

    required = set(candidate_keys()) if ensemble_weights is not None else {best_key}
    predictions = fit_full_candidate_predictions(train, test, y, required, cfg)
    best_path = cfg.output_dir / "submission_best_single.csv"
    make_submission(sample, test, predictions[best_key], best_path, cfg)
    output_paths = {"best_single": best_path}

    if ensemble_weights is not None:
        cube = np.stack([predictions[key] for key in candidate_keys()], axis=1)
        ensemble_scores = apply_rank_weights(cube, ensemble_weights)
        ensemble_path = cfg.output_dir / "submission_rank_ensemble.csv"
        make_submission(sample, test, ensemble_scores, ensemble_path, cfg)
        output_paths["rank_ensemble"] = ensemble_path
        np.savez_compressed(
            cfg.output_dir / "test_candidate_predictions.npz",
            identifiers=test[cfg.id_column].astype(str).to_numpy(dtype=str),
            scores=cube,
            candidates=np.asarray(candidate_keys(), dtype=str),
            labels=np.asarray(cfg.label_columns, dtype=str),
            ensemble_scores=ensemble_scores,
        )
    return output_paths


def _load_parameter_overrides(cfg: RunConfig, path: Path | None) -> None:
    """Apply explicit model-parameter overrides from a local JSON file."""

    if path is None:
        return
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    allowed = {"linear_parameters", "mlp_parameters"}
    unexpected = sorted(set(payload).difference(allowed))
    if unexpected:
        raise ValueError(f"Unexpected parameter-override sections: {unexpected}")
    if "linear_parameters" in payload:
        cfg.linear_parameters = payload["linear_parameters"]
    if "mlp_parameters" in payload:
        cfg.mlp_parameters = payload["mlp_parameters"]


def validate_parameter_structure(cfg: RunConfig) -> None:
    """Confirm that every permitted candidate has all required parameter records."""

    for representation in REPRESENTATIONS:
        for family in ("Logistic Regression", "Linear SVM"):
            try:
                label_parameters = cfg.linear_parameters[representation][family]
            except KeyError as exc:
                raise ValueError(
                    f"Missing linear parameters for {representation} + {family}."
                ) from exc
            if set(label_parameters) != set(LABEL_COLUMNS):
                raise ValueError(
                    f"{representation} + {family} must define exactly the six labels."
                )
            for label, values in label_parameters.items():
                if "C" not in values or float(values["C"]) <= 0:
                    raise ValueError(f"A positive C is required for {family}, {label}.")
        if representation not in cfg.mlp_parameters:
            raise ValueError(f"Missing MLP parameters for {representation}.")
        required = {"hidden_layer_sizes", "alpha", "learning_rate_init"}
        missing = required.difference(cfg.mlp_parameters[representation])
        if missing:
            raise ValueError(f"MLP parameters for {representation} are missing {sorted(missing)}.")


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Execute loading, EDA, validation, blending and optional submission refits."""

    paths = resolve_data_paths(args)
    cfg = RunConfig(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        seed=args.seed,
        n_splits=args.folds,
        threshold=args.threshold,
        meta_fraction=args.meta_fraction,
        meta_audit_fraction=args.meta_audit_fraction,
        use_ensemble=args.ensemble,
        ensemble_weight_mode=args.ensemble_weight_mode,
        ensemble_search_iterations=args.ensemble_search_iterations,
        observed_kaggle_score=args.observed_kaggle_score,
        word_max_features=args.word_max_features,
        character_max_features=args.character_max_features,
        mlp_svd_components=args.mlp_svd_components,
        mlp_max_iter=args.mlp_max_iter,
        row_limit=args.row_limit,
        run_eda=not args.no_eda,
        create_submissions=not args.benchmark_only,
    )
    _load_parameter_overrides(
        cfg,
        Path(args.parameters_json).expanduser().resolve() if args.parameters_json else None,
    )
    cfg.validate()
    validate_parameter_structure(cfg)
    set_reproducible_seed(cfg.seed)
    prepare_output_directory(cfg.output_dir)
    (cfg.output_dir / "tables").mkdir(parents=True, exist_ok=True)
    save_json(cfg.output_dir / "run_config.json", asdict(cfg))
    export_hyperparameters(cfg)

    train = validate_training_schema(read_csv_robust(paths.train, "train"), cfg)
    train = deterministic_row_limit(train, cfg.row_limit, cfg.seed)
    test: pd.DataFrame | None = None
    sample: pd.DataFrame | None = None
    if paths.test is not None and paths.sample_submission is not None:
        test, sample = validate_competition_schema(
            read_csv_robust(paths.test, "test"),
            read_csv_robust(paths.sample_submission, "sample_submission"),
            cfg,
        )
    input_manifest = {
        "train": {"path": str(paths.train), "sha256": sha256_file(paths.train)},
        "test": (
            {"path": str(paths.test), "sha256": sha256_file(paths.test)}
            if paths.test is not None
            else None
        ),
        "sample_submission": (
            {
                "path": str(paths.sample_submission),
                "sha256": sha256_file(paths.sample_submission),
            }
            if paths.sample_submission is not None
            else None
        ),
        "training_shape": list(train.shape),
        "test_shape": list(test.shape) if test is not None else None,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "iterative_stratification": importlib.metadata.version(
                "iterative-stratification"
            ),
        },
        "mlp_weighting": "Native sample_weight; scikit-learn 1.7 or later required",
        "network_access_used": False,
    }
    save_json(cfg.output_dir / "input_manifest.json", input_manifest)

    artefacts: dict[str, Any] = {"config": cfg, "paths": paths}
    if cfg.run_eda:
        artefacts["eda"] = run_eda(train, cfg)

    y = train.loc[:, list(cfg.label_columns)].to_numpy(dtype=np.int8)
    groups = build_group_structure(train[cfg.text_column], y)
    duplicate_rows = int(np.sum(groups.group_sizes[groups.group_sizes > 1]))
    label_signature = np.sum(
        y.astype(np.uint16) * (1 << np.arange(len(cfg.label_columns), dtype=np.uint16)),
        axis=1,
    )
    signature_counts = pd.Series(label_signature).groupby(groups.row_group).nunique()
    inconsistent_groups = int(signature_counts.gt(1).sum())
    save_json(
        cfg.output_dir / "duplicate_group_diagnostics.json",
        {
            "rows": len(train),
            "normalised_text_groups": len(groups.group_keys),
            "rows_in_non_singleton_groups": duplicate_rows,
            "groups_with_inconsistent_label_vectors": inconsistent_groups,
            "canonicalisation": "Unicode NFKC, case folding and whitespace collapse",
        },
    )
    plan = build_split_plan(groups, y, cfg)
    oof_scores, fold_results, summary, per_label, best_key = run_cross_validation(
        train, y, plan, cfg
    )
    artefacts.update(
        {
            "oof_scores": oof_scores,
            "fold_results": fold_results,
            "summary": summary,
            "per_label_results": per_label,
            "best_single_candidate": best_key,
        }
    )

    ensemble_weights: np.ndarray | None = None
    ensemble_audit_auc: float | None = None
    best_single_audit_auc: float | None = None
    if cfg.use_ensemble:
        (
            learned_weights,
            ensemble_accepted,
            ensemble_audit_auc,
            best_single_audit_auc,
        ) = run_meta_ensemble(train, y, plan, best_key, cfg)
        artefacts["ensemble_weights"] = learned_weights
        artefacts["ensemble_accepted_for_submission"] = ensemble_accepted
        if ensemble_accepted:
            ensemble_weights = learned_weights

    submission_paths: dict[str, Path] = {}
    if cfg.create_submissions:
        if test is None or sample is None:
            LOGGER.warning(
                "No test and sample-submission pair was supplied; submission refits were skipped."
            )
        else:
            submission_paths = create_final_submissions(
                train,
                test,
                sample,
                y,
                best_key,
                ensemble_weights,
                cfg,
            )
            artefacts["submission_paths"] = submission_paths

    save_json(
        cfg.output_dir / "run_summary.json",
        {
            "status": "completed",
            "model_families": list(MODEL_FAMILIES),
            "representations": list(REPRESENTATIONS),
            "candidate_count": len(candidate_keys()),
            "model_selection_rationale": MODEL_SELECTION_RATIONALE,
            "best_single_candidate": best_key,
            "selected_cross_validation_mean_column_roc_auc": float(
                summary.iloc[0]["fold_roc_auc_mean"]
            ),
            "selection_score_note": (
                "This is a model-selection estimate averaged across development "
                "folds, not an untouched final generalisation estimate."
            ),
            "leaderboard_comparison": {
                "user_provided_reference_auc": cfg.leaderboard_reference_auc,
                "observed_kaggle_score": cfg.observed_kaggle_score,
                "gap_to_reference": (
                    cfg.leaderboard_reference_auc - cfg.observed_kaggle_score
                    if cfg.observed_kaggle_score is not None
                    else None
                ),
                "status": (
                    "Post-submission comparison supplied by the user."
                    if cfg.observed_kaggle_score is not None
                    else "Pending an actual Kaggle submission; CV AUC is not compared with leaderboard AUC."
                ),
            },
            "ensemble_enabled": cfg.use_ensemble,
            "ensemble_audit": {
                "rank_ensemble_mean_column_roc_auc": ensemble_audit_auc,
                "best_single_mean_column_roc_auc": best_single_audit_auc,
                "accepted_for_submission": (
                    ensemble_weights is not None if cfg.use_ensemble else None
                ),
            },
            "submissions": {key: str(value) for key, value in submission_paths.items()},
        },
    )
    LOGGER.info("Experiment completed. Outputs are in %s.", cfg.output_dir)
    return artefacts


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line interface using British English descriptions."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exactly three toxic-comment classifiers with word TF-IDF and "
            "a binary character bag of n-grams. The script uses local files only "
            "and performs no downloads."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory searched recursively for the competition CSV or compressed files.",
    )
    parser.add_argument("--train", help="Explicit path to the labelled training CSV, GZIP or ZIP.")
    parser.add_argument("--test", help="Explicit path to the optional test CSV, GZIP or ZIP.")
    parser.add_argument(
        "--sample-submission",
        help="Explicit path to the optional sample-submission CSV, GZIP or ZIP.",
    )
    parser.add_argument(
        "--output-dir",
        default=f"toxic_comment_three_model_outputs_{time.strftime('%Y%m%d_%H%M%S')}",
        help="Directory for tables, figures, predictions, configuration and submissions.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Reproducible pseudo-random seed.")
    parser.add_argument(
        "--folds", type=int, default=3, help="Number of group-aware iterative multilabel CV folds."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fixed threshold used for accuracy, precision, recall and F1 metrics.",
    )
    parser.add_argument(
        "--meta-fraction",
        type=float,
        default=0.16,
        help="Share of duplicate groups reserved from development CV for ensemble work.",
    )
    parser.add_argument(
        "--meta-audit-fraction",
        type=float,
        default=0.50,
        help="Share of the meta-holdout kept untouched for ensemble audit.",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help=(
            "Opt in to separately audited rank blending. The strict assessment "
            "default produces only the best single algorithm."
        ),
    )
    parser.add_argument(
        "--ensemble-weight-mode",
        choices=("global", "per-label"),
        default="global",
        help="Use one common weight vector or separate weights for each toxicity label.",
    )
    parser.add_argument(
        "--ensemble-search-iterations",
        type=int,
        default=384,
        help="Number of seeded Dirichlet candidates added to deterministic blend weights.",
    )
    parser.add_argument(
        "--word-max-features",
        type=int,
        default=250_000,
        help="Maximum vocabulary size for word TF-IDF.",
    )
    parser.add_argument(
        "--character-max-features",
        type=int,
        default=300_000,
        help="Maximum vocabulary size for the binary character bag of n-grams.",
    )
    parser.add_argument(
        "--mlp-svd-components",
        type=int,
        default=192,
        help="Fold-local SVD dimension supplied to the compact CPU-oriented MLP.",
    )
    parser.add_argument(
        "--mlp-max-iter",
        type=int,
        default=40,
        help="Fixed training epochs for the group-safe MLP fit.",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        help="Optional reproducible row limit for a local smoke run; omit for final results.",
    )
    parser.add_argument(
        "--parameters-json",
        help="Optional local JSON overriding only linear_parameters and mlp_parameters.",
    )
    parser.add_argument(
        "--observed-kaggle-score",
        type=float,
        help=(
            "Optional score from a completed Kaggle submission. It is recorded only "
            "for post-submission comparison and never used for tuning."
        ),
    )
    parser.add_argument(
        "--no-eda", action="store_true", help="Skip exploratory tables and figures."
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run validation and optional blending without refitting submission models.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed progress messages."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the experiment and return a shell exit status."""

    parser = build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    try:
        run_pipeline(args)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        LOGGER.error("Experiment stopped: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
