import torch


class ContrastLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature


class PairSoftmaxContrastLoss(ContrastLoss):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature
        self._cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        sim_matrix = torch.cosine_similarity(text_embeddings.unsqueeze(1), text_pos_embeddings.unsqueeze(0), dim=-1)
        sim_matrix = sim_matrix / self.temperature
        labels = torch.arange(sim_matrix.size(0), device=text_embeddings.device, dtype=torch.long)
        loss = self._cross_entropy_loss(sim_matrix, labels)
        return loss


class PairSigmoidContrastLoss(ContrastLoss):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = text_embeddings.size(0)
        sim_matrix = torch.cosine_similarity(text_embeddings.unsqueeze(1), text_pos_embeddings.unsqueeze(0), dim=-1)
        sim_matrix = sim_matrix / self.temperature
        sim_matrix_diag = sim_matrix.diag()

        sim_diff_matrix = sim_matrix_diag.unsqueeze(1) - sim_matrix
        diag_mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        sim_diff_matrix = sim_diff_matrix.masked_fill(diag_mask, 1e9)

        loss = -torch.log(torch.sigmoid(sim_diff_matrix)).sum() / (batch_size**2 - batch_size)
        return loss


class TripletSoftmaxContrastLoss(ContrastLoss):
    def __init__(self, temperature: float = 0.05, add_swap_loss: bool = False):
        super().__init__()
        self.temperature = temperature
        self.add_swap_loss = add_swap_loss
        if self.add_swap_loss:
            self._pair_contrast_softmax_loss = PairSoftmaxContrastLoss(temperature)
        else:
            self._pair_contrast_softmax_loss = None

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
        text_neg_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        sim_pos_vector = torch.cosine_similarity(text_embeddings, text_pos_embeddings, dim=-1)
        sim_neg_matrix = torch.cosine_similarity(text_embeddings.unsqueeze(1), text_neg_embeddings.unsqueeze(0), dim=-1)
        sim_matrix = torch.cat([sim_pos_vector.unsqueeze(1), sim_neg_matrix], dim=1)
        sim_matrix = sim_matrix / self.temperature
        labels = torch.zeros(sim_matrix.size(0), dtype=torch.long, device=sim_matrix.device)
        loss = torch.nn.CrossEntropyLoss()(sim_matrix, labels)
        if self._pair_contrast_softmax_loss:
            loss += self._pair_contrast_softmax_loss(text_pos_embeddings, text_embeddings)
        return loss


class TripletSigmoidContrastLoss(ContrastLoss):
    def __init__(self, temperature: float = 0.05, add_swap_loss: bool = False):
        super().__init__()
        self.temperature = temperature
        self.add_swap_loss = add_swap_loss
        if self.add_swap_loss:
            self._pair_contrast_sigmoid_loss = PairSigmoidContrastLoss(temperature)
        else:
            self._pair_contrast_sigmoid_loss = None

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_pos_embeddings: torch.Tensor,
        text_neg_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        sim_pos_vector = torch.cosine_similarity(text_embeddings, text_pos_embeddings, dim=-1)
        sim_neg_matrix = torch.cosine_similarity(text_embeddings.unsqueeze(1), text_neg_embeddings.unsqueeze(0), dim=-1)
        sim_diff_matrix = sim_pos_vector.unsqueeze(1) - sim_neg_matrix
        sim_diff_matrix = sim_diff_matrix / self.temperature
        loss = -torch.log(torch.sigmoid(sim_diff_matrix)).mean()
        if self._pair_contrast_sigmoid_loss:
            loss += self._pair_contrast_sigmoid_loss(text_pos_embeddings, text_embeddings)
        return loss
