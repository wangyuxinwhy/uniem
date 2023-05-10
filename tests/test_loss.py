from typing import cast

import pytest
import torch

from uniem.criteria import (
    PairSigmoidContrastLoss,
    PairSoftmaxContrastLoss,
    TripletSigmoidContrastLoss,
    TripletSoftmaxContrastLoss,
)


def naive_pair_contrast_softmax_loss(text_embeddings: torch.Tensor, text_pos_embeddings: torch.Tensor, temperature: float):
    batch_size = text_embeddings.size(0)
    all_scores = None

    similarity_fct = torch.nn.CosineSimilarity(dim=-1)
    for i in range(0, batch_size):
        anchor_emb = text_embeddings[i].unsqueeze(0)
        pos_emb = text_pos_embeddings[i].unsqueeze(0)
        cur_score = similarity_fct(anchor_emb, pos_emb) / temperature
        for j in range(0, batch_size):
            if i == j:
                continue

            one_neg_emb = text_pos_embeddings[j].unsqueeze(0)
            one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / temperature
            cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
        if all_scores is None:
            all_scores = cur_score.unsqueeze(0)
        else:
            all_scores = torch.cat([all_scores, cur_score.unsqueeze(0)], dim=0)
    all_scores = cast(torch.Tensor, all_scores)
    labels = torch.zeros(all_scores.size(0)).long().to(text_embeddings.device)
    loss = torch.nn.CrossEntropyLoss()(all_scores, labels)
    return loss


def naive_pair_contrast_sigmoid_loss(text_embeddings: torch.Tensor, text_pos_embeddings: torch.Tensor, temperature: float):
    batch_size = text_embeddings.size(0)
    all_scores = []

    similarity_fct = torch.nn.CosineSimilarity(dim=-1)
    for i in range(0, batch_size):
        anchor_emb = text_embeddings[i].unsqueeze(0)
        pos_emb = text_pos_embeddings[i].unsqueeze(0)
        cur_score = (similarity_fct(anchor_emb, pos_emb) / temperature).item()
        for j in range(0, batch_size):
            if i == j:
                continue

            one_neg_emb = text_pos_embeddings[j].unsqueeze(0)
            one_neg_score = (similarity_fct(anchor_emb, one_neg_emb) / temperature).item()
            all_scores.append(torch.tensor([cur_score, one_neg_score]))
    all_scores = torch.stack(all_scores, dim=0)
    labels = torch.zeros(all_scores.size(0)).long().to(text_embeddings.device)
    loss = torch.nn.CrossEntropyLoss()(all_scores, labels)
    return loss


def naive_triplet_contrast_sigmoid_loss(
    text_embeddings: torch.Tensor,
    text_pos_embeddings: torch.Tensor,
    text_neg_embeddings: torch.Tensor,
    temperature: float,
    add_swap_loss: bool = False,
):
    embeddings_query = text_embeddings
    embeddings_pos = text_pos_embeddings
    embeddings_neg = text_neg_embeddings

    num = len(embeddings_query)
    all_scores = []
    from torch import nn

    similarity_fct = nn.CosineSimilarity(dim=-1)
    for i in range(0, num):
        anchor_emb = embeddings_query[i].unsqueeze(0)
        pos_emb = embeddings_pos[i].unsqueeze(0)
        cur_score = (similarity_fct(anchor_emb, pos_emb) / temperature).item()

        for j in range(0, num):
            one_neg_emb = embeddings_neg[j].unsqueeze(0)
            one_neg_score = (similarity_fct(anchor_emb, one_neg_emb) / temperature).item()
            all_scores.append(torch.tensor([cur_score, one_neg_score]))
    all_scores = torch.stack(all_scores, dim=0)
    labels = torch.zeros(all_scores.size(0)).long().to(text_embeddings.device)
    loss = torch.nn.CrossEntropyLoss()(all_scores, labels)
    if add_swap_loss:
        all_another_scores = []
        for i in range(0, num):
            anchor_emb = embeddings_pos[i].unsqueeze(0)
            pos_emb = embeddings_query[i].unsqueeze(0)
            cur_score = (similarity_fct(anchor_emb, pos_emb) / temperature).item()

            for j in range(0, num):
                if i == j:
                    continue
                one_neg_emb = embeddings_query[j].unsqueeze(0)
                one_neg_score = (similarity_fct(anchor_emb, one_neg_emb) / temperature).item()
                all_another_scores.append(torch.tensor([cur_score, one_neg_score]))
        all_another_scores = torch.stack(all_another_scores, dim=0)
        labels_another = torch.zeros(all_another_scores.size(0)).long().to(embeddings_query.device)
        loss += nn.CrossEntropyLoss()(all_another_scores, labels_another)
    return loss


def naive_triplet_contrast_softmax_loss(
    text_embeddings: torch.Tensor,
    text_pos_embeddings: torch.Tensor,
    text_neg_embeddings: torch.Tensor,
    temperature: float,
    add_swap_loss: bool = False,
):
    embeddings_query = text_embeddings
    embeddings_pos = text_pos_embeddings
    embeddings_neg = text_neg_embeddings

    num = len(embeddings_query)
    all_scores = None
    from torch import nn

    similarity_fct = nn.CosineSimilarity(dim=-1)
    for i in range(0, num):
        anchor_emb = embeddings_query[i].unsqueeze(0)
        pos_emb = embeddings_pos[i].unsqueeze(0)
        cur_score = similarity_fct(anchor_emb, pos_emb) / temperature

        for j in range(0, num):
            one_neg_emb = embeddings_neg[j].unsqueeze(0)
            one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / temperature
            cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
        if all_scores is None:
            all_scores = cur_score.unsqueeze(0)
        else:
            all_scores = torch.cat([all_scores, cur_score.unsqueeze(0)], dim=0)

    all_scores = cast(torch.Tensor, all_scores)
    labels = torch.zeros(all_scores.size(0)).long().to(embeddings_query.device)
    loss = nn.CrossEntropyLoss()(all_scores, labels)
    if add_swap_loss:
        all_another_scores = None
        for i in range(0, num):
            anchor_emb = embeddings_pos[i].unsqueeze(0)
            pos_emb = embeddings_query[i].unsqueeze(0)
            cur_score = similarity_fct(anchor_emb, pos_emb) / temperature

            for j in range(0, num):
                if i == j:
                    continue
                one_neg_emb = embeddings_query[j].unsqueeze(0)
                one_neg_score = similarity_fct(anchor_emb, one_neg_emb) / temperature
                cur_score = torch.cat([cur_score, one_neg_score], dim=-1)
            if all_another_scores is None:
                all_another_scores = cur_score.unsqueeze(0)
            else:
                all_another_scores = torch.cat([all_another_scores, cur_score.unsqueeze(0)], dim=0)
        all_another_scores = cast(torch.Tensor, all_another_scores)
        labels_another = torch.zeros(all_another_scores.size(0)).long().to(embeddings_query.device)
        loss += nn.CrossEntropyLoss()(all_another_scores, labels_another)

    return loss


@pytest.mark.parametrize('temperature', [0.05, 0.1, 1])
def test_pair_contrast_loss(temperature: float):
    text_embeddings = torch.randn(10, 768)
    text_pos_embeddings = torch.randn(10, 768)

    loss = PairSigmoidContrastLoss(temperature)(text_embeddings, text_pos_embeddings)
    naive_loss = naive_pair_contrast_sigmoid_loss(text_embeddings, text_pos_embeddings, temperature)
    assert torch.allclose(loss, naive_loss)

    loss = PairSoftmaxContrastLoss(temperature)(text_embeddings, text_pos_embeddings)
    naive_loss = naive_pair_contrast_softmax_loss(text_embeddings, text_pos_embeddings, temperature)
    assert torch.allclose(loss, naive_loss)


@pytest.mark.parametrize(
    'temperature, add_swap_loss', [(0.05, False), (0.1, False), (1, False), (0.05, True), (0.1, True), (1, True)]
)
def test_triplet_contrast_loss(temperature: float, add_swap_loss: bool):
    text_embeddings = torch.randn(10, 768)
    text_pos_embeddings = torch.randn(10, 768)
    text_neg_embeddings = torch.randn(10, 768)

    loss = TripletSoftmaxContrastLoss(temperature, add_swap_loss)(text_embeddings, text_pos_embeddings, text_neg_embeddings)
    naive_loss = naive_triplet_contrast_softmax_loss(
        text_embeddings, text_pos_embeddings, text_neg_embeddings, temperature, add_swap_loss
    )
    assert torch.allclose(loss, naive_loss)

    loss = TripletSigmoidContrastLoss(temperature, add_swap_loss)(text_embeddings, text_pos_embeddings, text_neg_embeddings)
    naive_loss = naive_triplet_contrast_sigmoid_loss(
        text_embeddings, text_pos_embeddings, text_neg_embeddings, temperature, add_swap_loss
    )
    assert torch.allclose(loss, naive_loss)
