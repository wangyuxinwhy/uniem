from dataclasses import asdict

from uniem.finetuner import FineTuner

from tests import FIXTURES_DIR


def test_finetuner(triplet_records, tmpdir):
    fintuner = FineTuner(
        model_name_or_path=str(FIXTURES_DIR / 'model'),
        dataset=[asdict(record) for record in triplet_records] * 10,
    )

    fintuner.run(batch_size=3, output_dir=tmpdir)

    assert (tmpdir / 'model').exists()
