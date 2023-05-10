from uniem.data import TripletCollator
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
