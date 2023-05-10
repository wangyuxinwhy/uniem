import pytest

from tests import FIXTURES_DIR
from uniem.data import MediDataset, TripletCollator
from uniem.data_structures import TripletRecord


def test_triplet_collator(tokenizer):
    records = [
        TripletRecord(
            text='I like apples',
            text_pos='I like oranges',
            text_neg='I want to eat apples',
        ),
        TripletRecord(
            text='I like football',
            text_pos='I like basketball',
            text_neg='I am a football player',
        ),
    ]
    collator = TripletCollator(tokenizer, max_length=10)

    batch = collator(records)

    assert set(batch.keys()) == {'text_ids', 'text_pos_ids', 'text_neg_ids'}
    assert batch['text_ids'].size(0) == 2


@pytest.mark.parametrize('batch_size', [4, 6, 8])
def test_medi_dataset(batch_size: int):
    dataset = MediDataset(FIXTURES_DIR / 'mini_medi.json', batch_size=batch_size, join_with='\n')

    for records in dataset:
        prompt = records[0].text.split(dataset.join_with, 1)[0]
        pos_prompt = records[0].text_pos.split(dataset.join_with, 1)[0]
        neg_prompt = records[0].text_neg.split(dataset.join_with, 1)[0]
        for record in records:
            assert record.text.startswith(prompt)
            assert record.text_pos.startswith(pos_prompt)
            assert record.text_neg.startswith(neg_prompt)
        assert len(set(record.text for record in records)) != 1
        assert len(set(record.text_pos for record in records)) != 1
        assert len(set(record.text_neg for record in records)) != 1
