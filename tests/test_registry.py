"""Unit tests for oilbn.data.registry.DataDictionary."""

from __future__ import annotations

from datetime import date

from oilbn.data.loaders import SeriesMetadata
from oilbn.data.registry import DataDictionary


def _metadata(name: str, category: str) -> SeriesMetadata:
    return SeriesMetadata(
        name=name,
        category=category,
        source="Test Source",
        frequency="daily",
        description=f"Description for {name}",
        units="USD",
        start_date=date(2020, 1, 1),
        end_date=date(2020, 12, 31),
    )


class TestDataDictionary:
    def test_empty_dictionary_has_expected_columns(self) -> None:
        registry = DataDictionary()
        frame = registry.to_frame()
        assert len(registry) == 0
        assert list(frame.columns) == [
            "name",
            "category",
            "source",
            "frequency",
            "units",
            "start_date",
            "end_date",
            "description",
        ]

    def test_register_and_to_frame(self) -> None:
        registry = DataDictionary()
        registry.register(_metadata("wti_futures", "financial"))
        registry.register(_metadata("fed_funds_rate", "macroeconomic"))

        assert len(registry) == 2
        assert "wti_futures" in registry

        frame = registry.to_frame()
        assert set(frame["name"]) == {"wti_futures", "fed_funds_rate"}
        # sorted by category then name: macroeconomic before financial alphabetically? verify order
        assert list(frame.sort_values("category")["category"]) == sorted(frame["category"])

    def test_replacing_entry_with_same_name_overwrites(self) -> None:
        registry = DataDictionary()
        registry.register(_metadata("wti_futures", "financial"))
        updated = _metadata("wti_futures", "financial")
        registry.register(updated)
        assert len(registry) == 1

    def test_to_latex_excludes_description_and_includes_name(self) -> None:
        registry = DataDictionary()
        registry.register(_metadata("wti_futures", "financial"))
        latex = registry.to_latex()
        assert r"wti\_\allowbreak futures" in latex
        assert "longtable" in latex

    def test_to_latex_escapes_underscores(self) -> None:
        registry = DataDictionary()
        registry.register(_metadata("fed_funds_rate", "macroeconomic"))
        latex = registry.to_latex()
        assert "fed_funds_rate" not in latex
        assert r"fed\_\allowbreak funds\_\allowbreak rate" in latex
