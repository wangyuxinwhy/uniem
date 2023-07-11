try:
    from functools import cached_property

    import torch
    from sentence_transformers import SentenceTransformer

    from uniem.utils import create_attention_mask_from_input_ids

    class SentenceTransformerWrapper(SentenceTransformer):
        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
            if attention_mask is None:
                attention_mask = create_attention_mask_from_input_ids(input_ids, self.pad_token_id)
            output = super().forward({'input_ids': input_ids, 'attention_mask': attention_mask})
            return output['sentence_embedding']

        @cached_property
        def pad_token_id(self) -> int:
            if hasattr(self.tokenizer, 'pad_token_id'):
                pad_token_id = self.tokenizer.pad_token_id   # type: ignore
                if isinstance(pad_token_id, int):
                    return pad_token_id

            config = self._first_module().config
            if hasattr(config, 'pad_token_id'):
                pad_token_id = config.pad_token_id   # type: ignore
                if isinstance(pad_token_id, int):
                    return pad_token_id

            return 0

except ImportError:
    raise ImportError('sentence_transformers is not installed. Please install it with "pip install sentence_transformers"')
