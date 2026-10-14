"""Small LaTeX text-escaping helper used when embedding arbitrary strings
(series names, descriptions, cleaning-report prose) into generated ``.tex``
fragments (data dictionary tables, cleaning narratives, etc.).
"""

from __future__ import annotations

import re

__all__ = ["latex_escape"]

_SPECIAL_CHARACTERS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_PATTERN = re.compile("|".join(re.escape(key) for key in _SPECIAL_CHARACTERS))


def latex_escape(text: str) -> str:
    """Escape LaTeX special characters in ``text`` so it is safe to embed
    outside of math mode (e.g. inside a ``longtable`` cell or report prose).

    Non-string input is returned unchanged, since pandas table columns may
    be mixed-type (e.g. a ``start_date`` column of ``datetime.date``).
    """
    if not isinstance(text, str):
        return text
    return _PATTERN.sub(lambda match: _SPECIAL_CHARACTERS[match.group()], text)
