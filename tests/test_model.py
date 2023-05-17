import pytest
import torch

from tests import FIXTURES_DIR
from uniem.model import (
    AutoEmbedder,
    EmbedderForPairTrain,
    EmbedderForTripletTrain,
    FirstLastEmbedder,
    LastMeanEmbedder,
    LastWeightedEmbedder,
    UniEmbedder,
    creat_mask_from_input_ids,
    mean_pooling,
)


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


def test_mean_pooling():
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

    pooled = mean_pooling(hidden_states, mask)

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
    model1 = EmbedderForTripletTrain(model_name_or_path=str(FIXTURES_DIR / 'model'), temperature=0.05, use_sigmoid=use_sigmoid)
    model2 = EmbedderForTripletTrain(
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

    loss1 = model1(**records)['loss']
    loss2 = model2(**records)['loss']
    assert isinstance(loss1, torch.Tensor)
    assert torch.allclose(loss1, loss2)


@pytest.mark.parametrize('use_sigmoid', [True, False])
def test_uniem_pair_model(use_sigmoid: bool):
    model = EmbedderForPairTrain(
        model_name_or_path=str(FIXTURES_DIR / 'model'),
        temperature=0.05,
        use_sigmoid=use_sigmoid,
    )
    records = {
        'text_ids': torch.tensor([[101, 2769, 1599, 3614, 4334, 4347, 3425, 102], [101, 2769, 1599, 3614, 6639, 4413, 102, 0]]),
        'text_pos_ids': torch.tensor([[101, 2769, 1599, 3614, 3580, 2094, 102], [101, 2769, 1599, 3614, 5074, 4413, 102]]),
    }

    loss = model(**records)['loss']
    assert isinstance(loss, torch.Tensor)


def test_last_weighted_embedder(transformers_model):
    embedder = LastWeightedEmbedder(transformers_model, pad_token_id=0)
    text_ids = torch.tensor([[101, 2769, 1599, 102], [101, 3614, 102, 0]])
    last_hidden_state = transformers_model(text_ids).last_hidden_state
    embeddings_0 = (
        (1 / 10) * last_hidden_state[0, 0, :]
        + (2 / 10) * last_hidden_state[0, 1, :]
        + (3 / 10) * last_hidden_state[0, 2, :]
        + (4 / 10) * last_hidden_state[0, 3, :]
    )
    embeddings_1 = (
        (1 / 6) * last_hidden_state[1, 0, :] + (2 / 6) * last_hidden_state[1, 1, :] + (3 / 6) * last_hidden_state[1, 2, :]
    )

    embeddings = embedder(text_ids)

    assert torch.allclose(embeddings[0], embeddings_0)
    assert torch.allclose(embeddings[1], embeddings_1)


@pytest.mark.parametrize('embedder_cls', [LastMeanEmbedder, FirstLastEmbedder])
def test_auto_embedder(transformers_model, tmpdir, embedder_cls):
    embedder = embedder_cls(transformers_model)

    embedder.save_pretrained(tmpdir)
    new_embedder = AutoEmbedder.from_pretrained(tmpdir)

    assert isinstance(new_embedder, embedder_cls)
    assert torch.allclose(embedder(torch.tensor([[1, 2, 3]])), new_embedder(torch.tensor([[1, 2, 3]])))


def test_uni_embedder():
    uni_embdder = UniEmbedder.from_pretrained(str(FIXTURES_DIR / 'model'))
    sentences = ['祖国万岁', 'Long live the motherland', '祖国万岁']

    embeddings = uni_embdder.encode(sentences)

    assert len(embeddings) == 3
    assert not torch.allclose(torch.from_numpy(embeddings[0]), torch.from_numpy(embeddings[1]))
    assert torch.allclose(torch.from_numpy(embeddings[0]), torch.from_numpy(embeddings[2]))
