"""
dataset_loader.py
-----------------
Loads any braindecode dataset whose configuration lives under the `datasets:`
key of configs/dataset_config.yaml.

Expected project layout
-----------------------
project_root/
├── configs/
│   └── dataset_config.yaml      ← all dataset entries live here
└── modules/
    ├── __init__.py
    └── dataset_loader.py        ← this file

Usage
-----
    from modules.dataset_loader import DatasetLoader

    # Load the NMT dataset (key must exist in dataset_config.yaml)
    loader = DatasetLoader(dataset_name="nmt")
    dataset = loader.load()
    loader.summary()

    # Load a different dataset without changing any code
    loader = DatasetLoader(dataset_name="tuab")
    dataset = loader.load()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


# ── known braindecode dataset classes ─────────────────────────────────────────
# Maps the `braindecode_class` string in the YAML to the actual class.
# Add new entries here as you add datasets to the config.
def _get_braindecode_registry() -> Dict[str, Any]:
    registry: Dict[str, Any] = {}
    try:
        from braindecode.datasets import NMT
        registry["NMT"] = NMT
    except ImportError:
        pass
    try:
        from braindecode.datasets import TUHAbnormal
        registry["TUHAbnormal"] = TUHAbnormal
    except ImportError:
        pass
    try:
        from braindecode.datasets import TUH
        registry["TUH"] = TUH
    except ImportError:
        pass
    try:
        from braindecode.datasets import SleepPhysionet
        registry["SleepPhysionet"] = SleepPhysionet
    except ImportError:
        pass
    return registry


# ── standard keys consumed directly by the loader ────────────────────────────
# Everything else in the dataset config block is passed as **kwargs.
_STANDARD_KEYS = {"braindecode_class", "path", "target_name",
                  "preload", "recording_ids", "n_jobs"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_config(config_path: Optional[Union[str, Path]]) -> Path:
    """
    Resolve the YAML config file path.

    If *config_path* is not supplied the function walks up from this file's
    location until it finds a `configs/dataset_config.yaml` alongside a
    `modules/` directory.
    """
    if config_path is not None:
        resolved = Path(config_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Config file not found: {resolved}")
        return resolved

    search_start = Path(__file__).resolve().parent  # .../modules/
    for candidate_root in [search_start.parent, search_start]:
        candidate = candidate_root / "configs" / "dataset_config.yaml"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not find 'configs/dataset_config.yaml' relative to the "
        f"modules directory ({search_start}). "
        "Either place it there or pass `config_path` explicitly."
    )


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# ── main class ────────────────────────────────────────────────────────────────

class DatasetLoader:
    """
    Generic braindecode dataset loader driven entirely by YAML configuration.

    Parameters
    ----------
    dataset_name : str
        Key under ``datasets:`` in the YAML config (e.g. ``"nmt"``).
    config_path : str | Path | None
        Explicit path to the YAML file. Auto-discovered when *None*.

    Attributes
    ----------
    cfg : dict
        The raw config block for the selected dataset.
    dataset : braindecode BaseConcatDataset | None
        Populated after calling :meth:`load`.

    Examples
    --------
    >>> loader = DatasetLoader("nmt")
    >>> ds = loader.load()
    >>> loader.summary()

    >>> # Switch datasets without touching any code
    >>> loader = DatasetLoader("tuab")
    >>> ds = loader.load()
    """

    def __init__(
        self,
        dataset_name: str,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._config_file = _find_config(config_path)
        logger.info("Using config: %s", self._config_file)

        raw_cfg = _load_yaml(self._config_file)

        if "datasets" not in raw_cfg:
            raise KeyError(
                f"Config file '{self._config_file}' must have a top-level "
                "'datasets' key."
            )

        available = list(raw_cfg["datasets"].keys())
        if dataset_name not in raw_cfg["datasets"]:
            raise KeyError(
                f"Dataset '{dataset_name}' not found in '{self._config_file}'. "
                f"Available datasets: {available}"
            )

        self.dataset_name: str = dataset_name
        self.cfg: Dict[str, Any] = raw_cfg["datasets"][dataset_name]
        self._registry = _get_braindecode_registry()

        self._validate_config()
        self.dataset = None

    # ── validation ────────────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        # Required keys
        for key in ("braindecode_class", "path", "target_name"):
            if key not in self.cfg:
                raise KeyError(
                    f"Dataset '{self.dataset_name}' config is missing "
                    f"required key '{key}' in '{self._config_file}'."
                )

        # Class must be registered
        cls_name = self.cfg["braindecode_class"]
        if cls_name not in self._registry:
            raise ValueError(
                f"braindecode_class '{cls_name}' is not recognised. "
                f"Registered classes: {list(self._registry.keys())}. "
                "Add it to _get_braindecode_registry() in dataset_loader.py."
            )

        # Path must exist
        data_path = Path(self.cfg["path"]).expanduser()
        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset path does not exist: {data_path}\n"
                f"Please update 'datasets.{self.dataset_name}.path' "
                f"in '{self._config_file}'."
            )

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def data_path(self) -> Path:
        return Path(self.cfg["path"]).expanduser().resolve()

    @property
    def target_name(self) -> str:
        return self.cfg["target_name"]

    @property
    def recording_ids(self) -> Optional[List[int]]:
        return self.cfg.get("recording_ids", None)

    @property
    def preload(self) -> bool:
        return bool(self.cfg.get("preload", False))

    @property
    def n_jobs(self) -> int:
        return int(self.cfg.get("n_jobs", 1))

    @property
    def extra_kwargs(self) -> Dict[str, Any]:
        """All config keys beyond the standard ones, forwarded as **kwargs."""
        return {k: v for k, v in self.cfg.items() if k not in _STANDARD_KEYS}

    # ── public API ────────────────────────────────────────────────────────────

    def load(self):
        """
        Instantiate and return the braindecode dataset using parameters from
        the config file.

        Standard parameters (path, target_name, preload, recording_ids,
        n_jobs) are passed explicitly; any additional YAML keys are forwarded
        as **kwargs so dataset-specific options work without code changes.

        Returns
        -------
        braindecode BaseConcatDataset
        """
        cls_name = self.cfg["braindecode_class"]
        cls = self._registry[cls_name]

        logger.info(
            "Loading %s ('%s') | path='%s' | target='%s' | "
            "recording_ids=%s | preload=%s | n_jobs=%d | extra=%s",
            cls_name,
            self.dataset_name,
            self.data_path,
            self.target_name,
            self.recording_ids,
            self.preload,
            self.n_jobs,
            self.extra_kwargs,
        )

        self.dataset = cls(
            path=str(self.data_path),
            target_name=self.target_name,
            recording_ids=self.recording_ids,
            preload=self.preload,
            n_jobs=self.n_jobs,
            **self.extra_kwargs,
        )

        logger.info("Loaded %d recording(s).", len(self.dataset.datasets))
        self._log_class_distribution()
        return self.dataset

    def summary(self) -> None:
        """
        Print a concise summary of the loaded dataset.
        Requires :meth:`load` to have been called first.
        """
        if self.dataset is None:
            print("Dataset not loaded yet. Call .load() first.")
            return

        desc = self.dataset.description
        cls_name = self.cfg["braindecode_class"]

        print(f"\n{'='*52}")
        print(f"  {cls_name} — '{self.dataset_name}'")
        print(f"{'='*52}")
        print(f"  Config file   : {self._config_file}")
        print(f"  Data path     : {self.data_path}")
        print(f"  Target        : {self.target_name}")
        print(f"  Recordings    : {len(self.dataset.datasets)}")

        if self.target_name in desc.columns:
            print(f"\n  Class distribution ({self.target_name}):")
            print(desc[self.target_name].value_counts().to_string(header=False))

        if "age" in desc.columns:
            print(
                f"\n  Age  — mean: {desc['age'].mean():.1f}  "
                f"std: {desc['age'].std():.1f}  "
                f"range: [{desc['age'].min()}, {desc['age'].max()}]"
            )

        if "gender" in desc.columns:
            print(f"\n  Gender distribution:")
            print(desc["gender"].value_counts().to_string(header=False))

        print(f"{'='*52}\n")

    # ── private helpers ───────────────────────────────────────────────────────

    def _log_class_distribution(self) -> None:
        if self.dataset is None:
            return
        desc = self.dataset.description
        if self.target_name in desc.columns:
            counts = desc[self.target_name].value_counts().to_dict()
            logger.info("Class distribution (%s): %s", self.target_name, counts)