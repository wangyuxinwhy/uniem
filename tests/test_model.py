import pytest
import torch

from tests import FIXTURES_DIR
from uniem.model import MeanPooler, UniEmbeddingModelForPairTrain, UniEmbeddingModelForTripletTrain, creat_mask_from_input_ids


def test_creat_mask_from_input_ids():
    input_ids = torch.tensor(
        [
            [1, 2, 3],
            [1, 2, 0],
        ],
        dtype=torch.long,
    )

    mask = creat_mask_from_input_ids(input_ids, 0)

    assert torch.equal(
        mask,
        torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 0],
            ],
            dtype=torch.bool,
        ),
    )


def test_mean_pooler():
    pooler = MeanPooler()
    hidden_states = torch.tensor(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 2, 3], [5, 6, 7], [10, 11, 12]],
        ],
        dtype=torch.float,
    )
    mask = torch.tensor(
        [
            [1, 1, 1],
            [1, 1, 0],
        ],
        dtype=torch.bool,
    )

    pooled = pooler(hidden_states, mask)

    assert torch.allclose(
        pooled,
        torch.tensor(
            [
                [4, 5, 6],
                [3, 4, 5],
            ],
            dtype=torch.float,
        ),
    )


@pytest.mark.parametrize('use_sigmoid', [True, False])
def test_uniem_triplet_model(use_sigmoid: bool):
    model = UniEmbeddingModelForTripletTrain(
        model_name_or_path=str(FIXTURES_DIR / 'model'),
        temperature=0.05,
        use_sigmoid=use_sigmoid,
    )
    records = {
        'text_ids': torch.tensor([[101, 2769, 1599, 3614, 4334, 4347, 3425, 102], [101, 2769, 1599, 3614, 6639, 4413, 102, 0]]),
        'text_pos_ids': torch.tensor([[101, 2769, 1599, 3614, 3580, 2094, 102], [101, 2769, 1599, 3614, 5074, 4413, 102]]),
        'text_neg_ids': torch.tensor(
            [
                [101, 5741, 3362, 1469, 3580, 2094, 6963, 3221, 3717, 3362, 102],
                [101, 3449, 6205, 3221, 6639, 4413, 6817, 1220, 1447, 102, 0],
            ]
        ),
    }

    loss = model(**records)
    assert loss is not None


@pytest.mark.parametrize('use_sigmoid', [True, False])
def test_uniem_pair_model(use_sigmoid: bool):
    model = UniEmbeddingModelForPairTrain(
        model_name_or_path=str(FIXTURES_DIR / 'model'),
        temperature=0.05,
        use_sigmoid=use_sigmoid,
    )
    records = {
        'text_ids': torch.tensor([[101, 2769, 1599, 3614, 4334, 4347, 3425, 102], [101, 2769, 1599, 3614, 6639, 4413, 102, 0]]),
        'text_pos_ids': torch.tensor([[101, 2769, 1599, 3614, 3580, 2094, 102], [101, 2769, 1599, 3614, 5074, 4413, 102]]),
    }

    loss = model(**records)
    assert loss is not None
