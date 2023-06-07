from pathlib import Path
from typing import cast

import typer

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from uniem.types import DatasetDescription, UniemDataset


def load_cmrc2018():
    dataset_dict = load_dataset('cmrc2018')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'question': 'text', 'context': 'text_pos'})
    return dataset_dict


cmrc2018_description = DatasetDescription(
    name='cmrc2018',
    is_symmetric=False,
    domains=['百科'],
    instruction_type='问答',
)


def load_belle_2m():
    dataset_dict = load_dataset('BelleGroup/train_2M_CN')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'instruction': 'text', 'output': 'text_pos'})
    return dataset_dict


belle_2m_description = DatasetDescription(
    name='belle_2m',
    is_symmetric=False,
    domains=['百科'],
    instruction_type='',
)


def load_zhihu_kol():
    dataset_dict = load_dataset('wangrui6/Zhihu-KOL')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'INSTRUCTION': 'text', 'RESPONSE': 'text_pos'})
    return dataset_dict


zhihu_kol_description = DatasetDescription(
    name='zhihu_kol',
    is_symmetric=False,
    domains=['百科'],
    instruction_type='问答',
)


def load_hc3_chinese():
    dataset_dict = load_dataset('Hello-SimpleAI/HC3-Chinese', 'all')
    dataset_dict = cast(DatasetDict, dataset_dict)
    processed_records = []
    for dataset in dataset_dict.values():
        for record in dataset:
            question = record.pop('question')
            human_answers = record.pop('human_answers')
            chatgpt_answers = record.pop('chatgpt_answers')
            for answer in human_answers:
                new_record = {'text': question, 'text_pos': answer}
                new_record.update(record)
                processed_records.append(new_record)
            for answer in chatgpt_answers:
                new_record = {'text': question, 'text_pos': answer}
                new_record.update(record)
                processed_records.append(new_record)
    return DatasetDict(train=Dataset.from_list(processed_records))


hc3_chinese_description = DatasetDescription(
    name='hc3_chinese',
    is_symmetric=False,
    domains=['百科'],
    instruction_type='问答',
)


def load_wiki_atomic_edits():
    import string

    from datasets import concatenate_datasets

    letters_and_digits = set(string.ascii_letters + string.digits)

    dataset1 = load_dataset('wiki_atomic_edits', 'chinese_insertions')['train']  # type: ignore
    dataset2 = load_dataset('wiki_atomic_edits', 'chinese_deletions')['train']  # type: ignore
    dataset = concatenate_datasets([dataset1, dataset2])  # type: ignore

    def concat_words(words: list[str]):
        text = ''
        for word in words:
            if word[0] in letters_and_digits or word[-1] in letters_and_digits:
                word = ' ' + word + ' '
            text += word
        text = text.strip()
        text = text.replace('  ', ' ')
        return text

    def _process(example):
        return {
            'base_sentence': concat_words(example['base_sentence'].split(' ')),
            'edited_sentence': concat_words(example['edited_sentence'].split(' ')),
        }

    dataset = dataset.map(_process)
    return dataset


wiki_atomic_edis_description = DatasetDescription(
    name='wiki_atomic_edits',
    is_symmetric=True,
    domains=['百科'],
    instruction_type='相似',
)


def load_chatmed_consult():
    dataset_dict = load_dataset('michaelwzhu/ChatMed_Consult_Dataset')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'query': 'text', 'response': 'text_pos'})
    return dataset_dict


chatmed_consult_description = DatasetDescription(
    name='chatmed_consult',
    is_symmetric=False,
    domains=['医药'],
    instruction_type='问答',
)


def load_amazon_reviews():
    dataset_dict = load_dataset('amazon_reviews_multi', 'zh')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'review_title': 'text', 'review_body': 'text_pos'})
    return dataset_dict


amazon_reviews_description = DatasetDescription(
    name='amazon_reviews',
    is_symmetric=False,
    domains=['电商'],
    instruction_type='摘要',
)


def load_xlsum():
    dataset = load_dataset('csebuetnlp/xlsum', 'chinese_simplified')
    dataset1 = dataset.select_columns(['title', 'summary', 'id', 'url'])
    dataset2 = dataset.select_columns(['title', 'text', 'id', 'url'])
    dataset1 = dataset1.rename_columns({'title': 'text', 'summary': 'text_pos'})
    dataset2 = dataset2.rename_columns({'title': 'text', 'text': 'text_pos'})
    all_datasets = list(dataset1.values()) + list(dataset2.values())   # type: ignore
    dataset = concatenate_datasets(all_datasets)
    return dataset


xlsum_description = DatasetDescription(
    name='xlsum',
    is_symmetric=False,
    domains=['新闻'],
    instruction_type='摘要',
)


def load_mlqa():
    dataset_dict = load_dataset('mlqa', 'mlqa-translate-train.zh')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'question': 'text', 'context': 'text_pos'})
    return dataset_dict


mlqa_description = DatasetDescription(
    name='mlqa',
    is_symmetric=False,
    domains=['百科'],
    instruction_type='问答',
)


def load_ocnli():
    dataset_dict = load_dataset('clue', 'ocnli')
    dataset_dict = cast(DatasetDict, dataset_dict)
    del dataset_dict['test']
    dataset_dict = dataset_dict.filter(lambda x: x['label'] == 1)
    dataset_dict = dataset_dict.rename_columns({'sentence1': 'text', 'sentence2': 'text_pos'})
    return dataset_dict


ocnli_description = DatasetDescription(
    name='ocnli',
    is_symmetric=True,
    domains=['口语'],
    instruction_type='推理',
)


def load_bq():
    dataset_dict = load_dataset('shibing624/nli_zh', 'BQ')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.filter(lambda x: x['label'] == 1)
    dataset_dict = dataset_dict.rename_columns({'sentence1': 'text', 'sentence2': 'text_pos'})
    return dataset_dict


bq_description = DatasetDescription(
    name='bq',
    is_symmetric=True,
    domains=['金融'],
    instruction_type='相似',
)


def load_lcqmc():
    dataset_dict = load_dataset('shibing624/nli_zh', 'LCQMC')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.filter(lambda x: x['label'] == 1)
    dataset_dict = dataset_dict.rename_columns({'sentence1': 'text', 'sentence2': 'text_pos'})
    return dataset_dict


lcqmc_description = DatasetDescription(
    name='lcqmc',
    is_symmetric=True,
    domains=['口语'],
    instruction_type='相似',
)


def load_pawsx():
    dataset_dict = load_dataset('shibing624/nli_zh', 'PAWSX')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.filter(lambda x: x['label'] == 1)
    dataset_dict = dataset_dict.rename_columns({'sentence1': 'text', 'sentence2': 'text_pos'})
    return dataset_dict


pawsx_description = DatasetDescription(
    name='pawsx',
    is_symmetric=True,
    domains=['百科'],
    instruction_type='相似',
)


def load_webqa():
    dataset_dict = load_dataset('suolyer/webqa')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'input': 'text', 'output': 'text_pos'})
    return dataset_dict


webqa_description = DatasetDescription(
    name='webqa',
    is_symmetric=False,
    domains=['百科'],
    instruction_type='问答',
)


def load_csl():
    dataset_dict = load_dataset('neuclir/csl')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'title': 'text', 'abstract': 'text_pos'})
    return dataset_dict


csl_description = DatasetDescription(
    name='csl',
    is_symmetric=False,
    domains=['学术'],
    instruction_type='摘要',
)


def load_dureader_robust():
    dataset_dict = load_dataset('PaddlePaddle/dureader_robust')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'question': 'text', 'context': 'text_pos'})
    return dataset_dict


dureader_robust_description = DatasetDescription(
    name='dureader_robust',
    is_symmetric=False,
    domains=['百科'],
    instruction_type='问答',
)


def load_miracl():
    dataset_dict = load_dataset('miracl/miracl-corpus', 'zh')
    dataset_dict = cast(DatasetDict, dataset_dict)

    try:
        from zhconv import convert
    except ImportError:
        raise ImportError('Please install zhconv first: pip install zhconv')

    def to_zh_cn(batch):
        zh_cn_titles = []
        zh_cn_texts = []
        for title, text in zip(batch['title'], batch['text']):
            zh_cn_titles.append(convert(title, 'zh-cn'))
            zh_cn_texts.append(convert(text, 'zh-cn'))
        return {
            'title': zh_cn_titles,
            'text': zh_cn_texts,
        }

    dataset_dict = dataset_dict.map(to_zh_cn, batched=True)
    dataset_dict = dataset_dict.rename_columns({'title': 'text', 'text': 'text_pos'})
    return dataset_dict


miracl_description = DatasetDescription(
    name='miracl',
    is_symmetric=False,
    domains=['百科'],
    instruction_type='摘要',
)


def load_firefly():
    dataset_dict = load_dataset('YeungNLP/firefly-train-1.1M')
    if isinstance(dataset_dict, Dataset):
        dataset_dict = DatasetDict({'train': dataset_dict})
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.rename_columns({'input': 'text', 'target': 'text_pos'})
    return dataset_dict


firefly_description = DatasetDescription(
    name='firefly',
    is_symmetric=False,
    domains=['百科'],
    instruction_type='',
)


def load_alpaca_gpt4():
    dataset_dict = load_dataset('shibing624/alpaca-zh')
    if isinstance(dataset_dict, Dataset):
        dataset_dict = DatasetDict({'train': dataset_dict})
    dataset_dict = cast(DatasetDict, dataset_dict)

    def concat_instruction_and_input(batch):
        return {
            'text': [f'{instruction} {input}' for instruction, input in zip(batch['instruction'], batch['input'])],
        }

    dataset_dict = dataset_dict.map(concat_instruction_and_input, batched=True)
    dataset_dict = dataset_dict.rename_columns({'output': 'text_pos'})
    return dataset_dict


alpaca_gpt4_description = DatasetDescription(
    name='alpaca_gpt4',
    is_symmetric=False,
    domains=['百科'],
    instruction_type='',
)


ALL_DATASETS: list[UniemDataset] = [
    UniemDataset(load_cmrc2018, cmrc2018_description),
    UniemDataset(load_belle_2m, belle_2m_description),
    UniemDataset(load_zhihu_kol, zhihu_kol_description),
    UniemDataset(load_hc3_chinese, hc3_chinese_description),
    UniemDataset(load_wiki_atomic_edits, wiki_atomic_edis_description),
    UniemDataset(load_chatmed_consult, chatmed_consult_description),
    UniemDataset(load_amazon_reviews, amazon_reviews_description),
    UniemDataset(load_xlsum, xlsum_description),
    UniemDataset(load_mlqa, mlqa_description),
    UniemDataset(load_ocnli, ocnli_description),
    UniemDataset(load_bq, bq_description),
    UniemDataset(load_lcqmc, lcqmc_description),
    UniemDataset(load_pawsx, pawsx_description),
    UniemDataset(load_webqa, webqa_description),
    UniemDataset(load_csl, csl_description),
    UniemDataset(load_dureader_robust, dureader_robust_description),
    UniemDataset(load_miracl, miracl_description),
    UniemDataset(load_firefly, firefly_description),
    UniemDataset(load_alpaca_gpt4, alpaca_gpt4_description),
]


def main(output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)

    for uniem_dataset in ALL_DATASETS:
        processed_dataset_dir = output_dir / f'{uniem_dataset.description.name}.dataset'
        if processed_dataset_dir.exists():
            print(f'{processed_dataset_dir} exists, skip')
            continue

        dataset = uniem_dataset.load_fn()
        dataset.save_to_disk(processed_dataset_dir)


if __name__ == '__main__':
    typer.run(main)
