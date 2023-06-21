import pytest
from uniem.data_structures import RecordType, infer_record_type


@pytest.mark.parametrize(
    'record, expected_record_type',
    [
        pytest.param(
            {'text': 'I like apples', 'text_pos': 'I like oranges'},
            RecordType.PAIR,
            id='pair',
        ),
        pytest.param(
            {'text': 'I like apples', 'text_pos': 'I like oranges', 'source': 'wikipedia'},
            RecordType.PAIR,
            id='pair_with_extra_fields',
        ),
        pytest.param(
            {'text': 'I like apples', 'text_pos': 'I like oranges', 'text_neg': 'I want to eat apples'},
            RecordType.TRIPLET,
            id='triplet',
        ),
        pytest.param(
            {'sentence1': 'I like apples', 'sentence2': 'I like oranges', 'label': 1.0},
            RecordType.SCORED_PAIR,
            id='scored_pair',
        ),
    ],
)
def test_infer_record_type(record: dict, expected_record_type: RecordType):
    assert infer_record_type(record) == expected_record_type
