import pytest
from transformers import AutoModel, AutoTokenizer  # type: ignore
from uniem.data_structures import TripletRecord

from tests import FIXTURES_DIR


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained(str(FIXTURES_DIR / 'model'))


@pytest.fixture
def transformers_model():
    return AutoModel.from_pretrained(str(FIXTURES_DIR / 'model'))


@pytest.fixture
def triplet_records():
    return [
        TripletRecord(
            text='我喜欢苹果',
            text_pos='我喜欢橘子',
            text_neg='苹果和橘子都是水果',
        ),
        TripletRecord(
            text='我喜欢足球',
            text_pos='我喜欢篮球',
            text_neg='梅西是足球运动员',
        ),
    ]
