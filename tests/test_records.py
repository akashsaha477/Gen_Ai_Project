"""Record lists and the AAMI mapping must match docs/RULES.md §4 exactly."""

import pytest

from ecgaug.data import records as R


def test_record_lists_match_the_protocol():
    assert R.DS1 == (101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
                     124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230)  # fmt: skip
    assert R.DS2 == (100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
                     212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234)  # fmt: skip
    assert R.PACED == (102, 104, 107, 217)


def test_record_lists_partition_the_database():
    sets = [set(R.DS1), set(R.DS2), set(R.PACED)]
    assert [len(s) for s in sets] == [22, 22, 4]
    assert len(set.union(*sets)) == 48 == len(R.ALL_RECORDS)
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]


def test_aami_mapping_matches_the_protocol():
    assert R.AAMI == {
        "N": ("N", "L", "R", "e", "j"),
        "S": ("A", "a", "J", "S"),
        "V": ("V", "E"),
        "F": ("F",),
        "Q": ("/", "f", "Q"),
    }
    assert R.CLASSES == ("N", "S", "V", "F")
    assert R.CLASS_INDEX == {"N": 0, "S": 1, "V": 2, "F": 3}
    assert len(R.SYMBOL_TO_AAMI) == 15  # no symbol is mapped to two classes


def test_symbol_sets_are_consistent():
    assert set(R.SYMBOL_TO_AAMI) <= R.QRS_SYMBOLS
    assert not R.QRS_SYMBOLS & R.NON_BEAT_SYMBOLS
    assert "!" in R.QRS_SYMBOLS and "!" not in R.SYMBOL_TO_AAMI  # flutter wave: QRS-type, no AAMI class


def test_dataset_of():
    assert R.dataset_of(101) == "DS1" and R.dataset_of(100) == "DS2" and R.dataset_of(102) == "paced"
    with pytest.raises(ValueError):
        R.dataset_of(999)


def test_same_subject_pair_straddles_the_split():
    assert R.SAME_SUBJECT == ((201, 202),)
    assert 201 in R.DS1 and 202 in R.DS2
