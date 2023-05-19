from pathlib import Path
from typing import cast

import typer
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from uniem.types import DatasetDescription, UniemDataset


def load_cmrc2018(remove_other_colums: bool = False):
    dataset_dict = load_dataset("cmrc2018")
    dataset_dict = cast(DatasetDict, dataset_dict)
    column_names = dataset_dict.column_names['train']
    dataset_dict = dataset_dict.rename_columns({'question': 'text', 'context': 'text_pos'})
    if remove_other_colums:
        dataset_dict = dataset_dict.remove_columns(list(set(column_names) - set(['question', 'context'])))
    return dataset_dict


cmrc2018_description = DatasetDescription(
    name='cmrc2018',
    is_symmetric=False,
    domains=['百科'],
    raw_size=14363,
    instruction_type='问答',
)


def load_belle_2m(remove_other_colums: bool = False):
    dataset_dict = load_dataset('BelleGroup/train_2M_CN')
    dataset_dict = cast(DatasetDict, dataset_dict)
    column_names = dataset_dict.column_names['train']
    dataset_dict = dataset_dict.rename_columns({'instruction': 'text', 'output': 'text_pos'})
    if remove_other_colums:
        dataset_dict = dataset_dict.remove_columns(list(set(column_names) - set(['instruction', 'output'])))
    return dataset_dict


belle_2m_description = DatasetDescription(
    name='belle_2m',
    is_symmetric=False,
    domains=['百科'],
    raw_size=2000000,
    instruction_type='',
)


def load_zhihu_kol(remove_other_colums: bool = False):
    dataset_dict = load_dataset('wangrui6/Zhihu-KOL')
    dataset_dict = cast(DatasetDict, dataset_dict)
    column_names = dataset_dict.column_names['train']
    dataset_dict = dataset_dict.rename_columns({'INSTRUCTION': 'text', 'RESPONSE': 'text_pos'})
    if remove_other_colums:
        dataset_dict = dataset_dict.remove_columns(list(set(column_names) - set(['INSTRUCTION', 'RESPONSE'])))
    return dataset_dict


zhihu_kol_description = DatasetDescription(
    name='zhihu_kol',
    is_symmetric=False,
    domains=['百科'],
    raw_size=1006218,
    instruction_type='问答',
)

def load_hc3_chinese(remove_other_colums: bool = False):
    dataset_dict = load_dataset("Hello-SimpleAI/HC3-Chinese", "all")
    dataset_dict = cast(DatasetDict, dataset_dict)
    processed_records = []
    for dataset in dataset_dict.values():
        for record in dataset:
            question = record.pop('question')
            human_answers = record.pop('human_answers')
            chatgpt_answers = record.pop('chatgpt_answers')
            for answer in human_answers:
                new_record = {
                    'text': question,
                    'text_pos': answer
                }
                if not remove_other_colums:
                    new_record.update(record)
                processed_records.append(new_record)
            for answer in chatgpt_answers:
                new_record = {
                    'text': question,
                    'text_pos': answer
                }
                if not remove_other_colums:
                    new_record.update(record)
                processed_records.append(new_record)
    return DatasetDict(train=Dataset.from_list(processed_records))


hc3_chinese_description = DatasetDescription(
    name='hc3_chinese',
    is_symmetric=False,
    domains=['百科'],
    raw_size=39781,
    instruction_type='问答',
)

def load_wiki_atomic_edits(remove_other_colums: bool = False):
    import string
    from datasets import concatenate_datasets

    letters_and_digits = set(string.ascii_letters + string.digits)

    dataset1 = load_dataset("wiki_atomic_edits", "chinese_insertions")['train']  # type: ignore
    dataset2 = load_dataset("wiki_atomic_edits", "chinese_deletions")['train']  # type: ignore
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
            'edited_sentence': concat_words(example['edited_sentence'].split(' '))
        }
    dataset = dataset.map(_process)
    if remove_other_colums:
        dataset = dataset.remove_columns(['id', 'phrase'])
    return dataset


wiki_atomic_edis_description = DatasetDescription(
    name='wiki_atomic_edits',
    is_symmetric=True,
    domains=['百科'],
    raw_size=1213780,
    instruction_type='相似',
)


def load_chatmed_consult(remove_other_colums: bool = False):
    dataset_dict = load_dataset('michaelwzhu/ChatMed_Consult_Dataset')
    dataset_dict = cast(DatasetDict, dataset_dict)
    column_names = dataset_dict.column_names['train']
    dataset_dict = dataset_dict.rename_columns({'query': 'text', 'response': 'text_pos'})
    if remove_other_colums:
        dataset_dict = dataset_dict.remove_columns(list(set(column_names) - set(['query', 'response'])))
    return dataset_dict


chatmed_consult_description = DatasetDescription(
    name='chatmed_consult',
    is_symmetric=False,
    domains=['医药'],
    raw_size=549326,
    instruction_type='问答',
)

def load_amazon_reviews(remove_other_colums: bool = False):
    dataset_dict = load_dataset('amazon_reviews_multi', 'zh')
    dataset_dict = cast(DatasetDict, dataset_dict)
    column_names = dataset_dict.column_names['train']
    dataset_dict = dataset_dict.rename_columns({'review_title': 'text', 'review_body': 'text_pos'})
    if remove_other_colums:
        dataset_dict = dataset_dict.remove_columns(list(set(column_names) - set(['review_title', 'review_body'])))
    return dataset_dict


amazon_reviews_description = DatasetDescription(
    name='amazon_reviews',
    is_symmetric=False,
    domains=['电商'],
    raw_size=210000,
    instruction_type='摘要',
)


def load_xlsum(remove_other_colums: bool = False):
    dataset = load_dataset('csebuetnlp/xlsum', 'chinese_simplified')

    if remove_other_colums:
        dataset1 = dataset.select_columns(['title', 'summary'])
        dataset2 = dataset.select_columns(['title', 'text'])
    else:
        dataset1 = dataset.select_columns(['title', 'summary', 'id', 'url'])
        dataset2 = dataset.select_columns(['title', 'text', 'id', 'url'])
    dataset1 = dataset1.rename_columns({
        'title': 'text',
        'summary': 'text_pos'
    })
    dataset2 = dataset2.rename_columns({
        'title': 'text',
        'text': 'text_pos'
    })
    all_datasets = list(dataset1.values()) + list(dataset2.values()) # type: ignore
    dataset = concatenate_datasets(all_datasets)
    return dataset


xlsum_description = DatasetDescription(
    name='xlsum',
    is_symmetric=False,
    domains=['新闻'],
    raw_size=93404,
    instruction_type='摘要',
)

def load_mlqa(remove_other_colums: bool = False):
    dataset_dict = load_dataset('mlqa', 'mlqa-translate-train.zh')
    dataset_dict = cast(DatasetDict, dataset_dict)
    column_names = dataset_dict.column_names['train']
    dataset_dict = dataset_dict.rename_columns({'question': 'text', 'context': 'text_pos'})
    if remove_other_colums:
        dataset_dict = dataset_dict.remove_columns(list(set(column_names) - set(['question', 'context'])))
    return dataset_dict


mlqa_description = DatasetDescription(
    name='mlqa',
    is_symmetric=False,
    domains=['百科'],
    raw_size=85853,
    instruction_type='问答',
)


def load_ocnli(remove_other_colums: bool = False):
    dataset_dict = load_dataset('clue', 'ocnli')
    dataset_dict = cast(DatasetDict, dataset_dict)
    del dataset_dict['test']
    dataset_dict = dataset_dict.filter(lambda x: x['label'] == 1)
    column_names = dataset_dict.column_names['train']
    dataset_dict = dataset_dict.rename_columns({'sentence1': 'text', 'sentence2': 'text_pos'})
    if remove_other_colums:
        dataset_dict = dataset_dict.remove_columns(list(set(column_names) - set(['sentence1', 'sentence2'])))
    return dataset_dict


ocnli_description = DatasetDescription(
    name='ocnli',
    is_symmetric=True,
    domains=['口语'],
    raw_size=17726,
    instruction_type='推理',
)


def load_bq(remove_other_colums: bool = False):
    dataset_dict = load_dataset('shibing624/nli_zh', 'BQ')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.filter(lambda x: x['label'] == 1)
    column_names = dataset_dict.column_names['train']
    dataset_dict = dataset_dict.rename_columns({'sentence1': 'text', 'sentence2': 'text_pos'})
    if remove_other_colums:
        dataset_dict = dataset_dict.remove_columns(list(set(column_names) - set(['sentence1', 'sentence2'])))
    return dataset_dict


bq_description = DatasetDescription(
    name='bq',
    is_symmetric=True,
    domains=['金融'],
    raw_size=60000,
    instruction_type='相似',
)


def load_lcqmc(remove_other_colums: bool = False):
    dataset_dict = load_dataset('shibing624/nli_zh', 'LCQMC')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.filter(lambda x: x['label'] == 1)
    column_names = dataset_dict.column_names['train']
    dataset_dict = dataset_dict.rename_columns({'sentence1': 'text', 'sentence2': 'text_pos'})
    if remove_other_colums:
        dataset_dict = dataset_dict.remove_columns(list(set(column_names) - set(['sentence1', 'sentence2'])))
    return dataset_dict


lcqmc_description = DatasetDescription(
    name='lcqmc',
    is_symmetric=True,
    domains=['口语'],
    raw_size=149226,
    instruction_type='相似',
)


def load_pawsx(remove_other_colums: bool = False):
    dataset_dict = load_dataset('shibing624/nli_zh', 'PAWSX')
    dataset_dict = cast(DatasetDict, dataset_dict)
    dataset_dict = dataset_dict.filter(lambda x: x['label'] == 1)
    column_names = dataset_dict.column_names['train']
    dataset_dict = dataset_dict.rename_columns({'sentence1': 'text', 'sentence2': 'text_pos'})
    if remove_other_colums:
        dataset_dict = dataset_dict.remove_columns(list(set(column_names) - set(['sentence1', 'sentence2'])))
    return dataset_dict


pawsx_description = DatasetDescription(
    name='pawsx',
    is_symmetric=True,
    domains=['百科'],
    raw_size=23576,
    instruction_type='相似',
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
]

def main(output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for uniem_dataset in ALL_DATASETS:
        processed_dataset_dir = output_dir / f'{uniem_dataset.description.name}.dataset'
        if processed_dataset_dir.exists():
            print(f'{processed_dataset_dir} exists, skip')
            continue

        dataset = uniem_dataset.load_fn(False)
        dataset.save_to_disk(processed_dataset_dir)


if __name__ == '__main__':
    typer.run(main)
