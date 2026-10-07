"""MIT-BIH record lists and the AAMI label mapping (docs/RULES.md §4).

Inter-patient protocol of de Chazal et al. (2004): DS1 for training and validation, DS2 for test,
the four paced records excluded.
"""

from __future__ import annotations

FS = 360
LEAD = "MLII"

DS1: tuple[int, ...] = (
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
    124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230,
)  # fmt: skip
DS2: tuple[int, ...] = (
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
    212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234,
)  # fmt: skip
PACED: tuple[int, ...] = (102, 104, 107, 217)
ALL_RECORDS: tuple[int, ...] = tuple(sorted(DS1 + DS2 + PACED))

# Records from the same subject that fall in different sets (PhysioNet mitdbdir introduction).
SAME_SUBJECT: tuple[tuple[int, int], ...] = ((201, 202),)

CLASSES: tuple[str, ...] = ("N", "S", "V", "F")
CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}

AAMI: dict[str, tuple[str, ...]] = {
    "N": ("N", "L", "R", "e", "j"),
    "S": ("A", "a", "J", "S"),
    "V": ("V", "E"),
    "F": ("F",),
    "Q": ("/", "f", "Q"),
}
SYMBOL_TO_AAMI: dict[str, str] = {sym: cls for cls, syms in AAMI.items() for sym in syms}

# Annotation symbols that WFDB treats as QRS complexes (isqrs() in ecgmap.h: codes 1-13, 25, 30,
# 31, 34, 35, 38, 41). RR intervals are measured over all of them, including the ones that map to
# no AAMI class here ('!' flutter wave, 'B', '?', 'n', 'r').
QRS_SYMBOLS: frozenset[str] = frozenset(
    ["N", "L", "R", "a", "V", "F", "J", "A", "S", "E", "j", "/", "Q", "B", "?", "!", "e", "n", "f", "r"]
)
# Every other symbol in the WFDB annotation table. These are not beats and are ignored.
NON_BEAT_SYMBOLS: frozenset[str] = frozenset(
    [" ", "~", "|", "s", "T", "*", "D", '"', "=", "p", "^", "t", "+", "u", "[", "]", "@", "x", "(", ")"]
)


def dataset_of(record: int) -> str:
    """Return 'DS1', 'DS2' or 'paced' for an MIT-BIH record number."""
    record = int(record)
    if record in DS1:
        return "DS1"
    if record in DS2:
        return "DS2"
    if record in PACED:
        return "paced"
    raise ValueError(f"{record} is not an MIT-BIH Arrhythmia Database record")
