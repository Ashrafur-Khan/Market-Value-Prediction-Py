"""Configuration helpers for locating project resources."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    #Resolve the canonical project directories relative to this package

    project_root: Path
    data_dir: Path
    output_dir: Path
    market_value_file: Path
    trends_workbook: Path
    countries_workbook: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        #Auto-detect paths when running inside the repository tree
        package_dir = Path(__file__).resolve().parent
        project_root = package_dir.parent.parent
        data_dir = project_root / "data"
        output_dir = project_root / "outputs"
        return cls(
            project_root=project_root,
            data_dir=data_dir,
            output_dir=output_dir,
            market_value_file=data_dir / "MarketValueData_English.xlsx",
            trends_workbook=data_dir / "Trends_FactorConversion.xlsx",
            countries_workbook=data_dir / "Countries.xlsx",
        )


PATHS = ProjectPaths.discover()
