from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from winter_ortho.utils.paths import get_project_root, resolve_config_path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = resolve_config_path(config_path or "config/default.yaml")
    config = _load_yaml(path)
    root = get_project_root(path.parent)
    for key in ("raw_root", "intermediate_root", "output_root"):
        if key in config.get("paths", {}):
            config["paths"][key] = str(root / config["paths"][key])
    raw_paths = config.get("paths", {})
    for key in ("orthophoto", "dem", "winter_reference"):
        if key in raw_paths:
            raw_paths[key] = str(root / raw_paths[key])
    tlm = raw_paths.get("tlm3d", {})
    for key, value in tlm.items():
        tlm[key] = str(root / value)
    return config


def load_class_rules(rules_path: str | Path | None = None) -> dict[str, Any]:
    path = resolve_config_path(rules_path or "config/class_rules.yaml")
    return _load_yaml(path)


def load_profile(profile: str, profiles_dir: str | Path | None = None) -> dict[str, Any]:
    root = get_project_root()
    if profiles_dir:
        base = Path(profiles_dir)
    else:
        cwd_profile = Path.cwd() / "config" / "rendering_profiles"
        base = cwd_profile if cwd_profile.exists() else root / "config" / "rendering_profiles"
    path = base / f"{profile}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Rendering profile not found: {path}")
    return _load_yaml(path)
