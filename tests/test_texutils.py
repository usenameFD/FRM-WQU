"""Unit tests for oilbn.utils.texutils.latex_escape."""

from __future__ import annotations

from oilbn.utils.texutils import latex_escape


class TestLatexEscape:
    def test_escapes_underscore(self) -> None:
        assert latex_escape("wti_futures") == r"wti\_futures"

    def test_escapes_multiple_special_characters(self) -> None:
        assert latex_escape("50% & $x_1$") == r"50\% \& \$x\_1\$"

    def test_non_string_input_passed_through(self) -> None:
        assert latex_escape(42) == 42  # type: ignore[arg-type]
        assert latex_escape(None) is None  # type: ignore[arg-type]

    def test_string_without_special_characters_is_unchanged(self) -> None:
        assert latex_escape("plain text") == "plain text"
