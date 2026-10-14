"""Data dictionary registry (GWP1 Step 4).

Builds the table of "data used, frequency, source, start date, end date,
and other relevant fields" required by the assignment, directly from the
:class:`~oilbn.data.loaders.SeriesMetadata` objects produced by each loader.
"""

from __future__ import annotations

import pandas as pd

from oilbn.data.loaders import SeriesMetadata
from oilbn.utils.texutils import latex_escape

__all__ = ["DataDictionary"]


class DataDictionary:
    """Accumulates :class:`SeriesMetadata` entries into a reportable table."""

    def __init__(self) -> None:
        self._entries: dict[str, SeriesMetadata] = {}

    def register(self, metadata: SeriesMetadata) -> None:
        """Add (or replace) one series' metadata entry."""
        self._entries[metadata.name] = metadata

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __getitem__(self, name: str) -> SeriesMetadata:
        return self._entries[name]

    def to_frame(self) -> pd.DataFrame:
        """Return the data dictionary as a tidy, report-ready ``DataFrame``."""
        if not self._entries:
            return pd.DataFrame(
                columns=[
                    "name",
                    "category",
                    "source",
                    "frequency",
                    "units",
                    "start_date",
                    "end_date",
                    "description",
                ]
            )
        rows = [
            {
                "name": m.name,
                "category": m.category,
                "source": m.source,
                "frequency": m.frequency,
                "units": m.units,
                "start_date": m.start_date,
                "end_date": m.end_date,
                "description": m.description,
            }
            for m in self._entries.values()
        ]
        frame = pd.DataFrame(rows).sort_values(["category", "name"]).reset_index(drop=True)
        return frame

    def to_latex(self, caption: str = "Data dictionary", label: str = "tab:data-dictionary") -> str:
        """Render the dictionary as a LaTeX ``longtable`` for direct report inclusion.

        Text columns are LaTeX-escaped (series names/descriptions may
        contain underscores, e.g. ``wti_futures``) so the generated
        fragment compiles without manual post-editing.
        """
        frame = self.to_frame().drop(columns=["description"])
        text_columns = frame.select_dtypes(include=["object", "str"]).columns
        frame[text_columns] = frame[text_columns].map(latex_escape)
        # Series names are long, underscore-separated identifiers with no
        # natural break point for LaTeX's paragraph-mode column wrapping
        # (``p{...}``); insert an explicit break opportunity after each
        # escaped underscore so long names wrap instead of overflowing.
        frame["name"] = frame["name"].str.replace(r"\_", r"\_\allowbreak ", regex=False)
        frame = frame.rename(
            columns={
                "name": "Name",
                "category": "Category",
                "source": "Source",
                "frequency": "Freq.",
                "units": "Units",
                "start_date": "Start",
                "end_date": "End",
            }
        )
        return frame.to_latex(
            index=False,
            longtable=True,
            caption=caption,
            label=label,
            na_rep="--",
            column_format=(
                r">{\raggedright\arraybackslash}p{1.9cm}"
                r">{\raggedright\arraybackslash}p{2.3cm}"
                r">{\raggedright\arraybackslash}p{4.2cm}"
                r">{\raggedright\arraybackslash}p{1.4cm}"
                r">{\raggedright\arraybackslash}p{0.8cm}"
                r">{\raggedright\arraybackslash}p{1.45cm}"
                r">{\raggedright\arraybackslash}p{1.45cm}"
            ),
        )

    def to_csv(self, path: str) -> None:
        """Persist the data dictionary to CSV (e.g., ``data/processed/data_dictionary.csv``)."""
        self.to_frame().to_csv(path, index=False)
