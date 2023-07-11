from dataclasses import asdict
from typing import cast

from uniem.finetuner import FineTuner
from uniem.model import UniemEmbedder

from tests import FIXTURES_DIR


def test_finetuner(triplet_records, tmpdir):
    fintuner = FineTuner.from_pretrained(
        model_name_or_path=str(FIXTURES_DIR / 'model'),
        dataset=[asdict(record) for record in triplet_records] * 10,
        model_type='uniem',
    )

    embedder = fintuner.run(batch_size=3, output_dir=tmpdir)
    embedder = cast(UniemEmbedder, embedder)
    embedder.save_pretrained(tmpdir / 'model')

    assert (tmpdir / 'model').exists()
