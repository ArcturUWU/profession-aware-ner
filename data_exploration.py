import os
import re
from natasha import Segmenter, Doc
import pandas as pd


def load_brat_data(directory: str):
    """Load Brat-format files (.ann + .txt) into a list of dicts, handling discontinuous spans."""
    segmenter = Segmenter()
    records = []

    for filename in os.listdir(directory):
        if not filename.endswith('.txt'):
            continue
        base = os.path.splitext(filename)[0]
        txt_path = os.path.join(directory, base + '.txt')
        ann_path = os.path.join(directory, base + '.ann')

        with open(txt_path, encoding='utf-8') as f:
            text = f.read()

        # parse entities
        entities = []
        if os.path.exists(ann_path):
            with open(ann_path, encoding='utf-8') as f:
                for line in f:
                    if not line.startswith('T'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) != 3:
                        continue
                    _, tag_info, ent_text = parts
                    # split label and span info
                    try:
                        label, spans_str = tag_info.split(' ', 1)
                    except ValueError:
                        continue
                    # handle discontinuous spans
                    for span in spans_str.split(';'):
                        coords = span.strip().split()
                        if len(coords) != 2:
                            continue
                        start, end = map(int, coords)
                        entities.append({
                            'start': start,
                            'end': end,
                            'label': label,
                            'text': text[start:end]
                        })

        # tokenize with Natasha
        doc = Doc(text)
        doc.segment(segmenter)
        tokens = [token.text for token in doc.tokens]
        token_spans = [(token.start, token.stop) for token in doc.tokens]

        # assign tags
        tags = ['O'] * len(tokens)
        for ent in entities:
            started = False
            for i, (st, sp) in enumerate(token_spans):
                if st < ent['end'] and sp > ent['start']:
                    if not started:
                        tags[i] = 'B-' + ent['label']
                        started = True
                    else:
                        tags[i] = 'I-' + ent['label']

        # record doc-level
        records.append({
            'doc_id': base,
            'text': text,
            'tokens': tokens,
            'tags': tags,
            'entities': entities
        })

    return records


def records_to_token_df(records):
    """Convert records list into a flat DataFrame of tokens and tags."""
    rows = []
    for rec in records:
        for idx, (tok, tag) in enumerate(zip(rec['tokens'], rec['tags'])):
            rows.append({
                'doc_id': rec['doc_id'],
                'token_index': idx,
                'token': tok,
                'tag': tag
            })
    return pd.DataFrame(rows)


def records_to_entity_df(records):
    """Extract entity-level DataFrame from records."""
    rows = []
    for rec in records:
        for ent in rec['entities']:
            rows.append({
                'doc_id': rec['doc_id'],
                'label': ent['label'],
                'text': ent['text'],
                'start': ent['start'],
                'end': ent['end'],
                'length': ent['end'] - ent['start']
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    base_path = 'HSE/nerel_dataset/data/NEREL-v1.1'
    train_dir = os.path.join(base_path, 'train')
    dev_dir = os.path.join(base_path, 'dev')

    # load data
    print('Loading train data...')
    train_records = load_brat_data(train_dir)
    print(f'Total train documents: {len(train_records)}')

    print('Loading dev data...')
    dev_records = load_brat_data(dev_dir)
    print(f'Total dev documents: {len(dev_records)}')

    # build DataFrames
    train_tokens_df = records_to_token_df(train_records)
    dev_tokens_df = records_to_token_df(dev_records)

    train_entities_df = records_to_entity_df(train_records)
    dev_entities_df = records_to_entity_df(dev_records)

    # exploration
    print('\nToken-level distribution:')
    print(train_tokens_df['tag'].value_counts())

    print('\nEntity-level distribution:')
    print(train_entities_df['label'].value_counts())

    print('\nTop 10 longest entities:')
    print(train_entities_df.nlargest(10, 'length')[['text', 'label', 'length']])

    # overview stats
    print(f"Avg tokens per doc (train): {train_tokens_df.groupby('doc_id').size().mean():.2f}")
    print(f"Avg entities per doc (train): {train_entities_df.groupby('doc_id').size().mean():.2f}")

    # export summaries
    train_entities_df['label'].value_counts().to_csv('train_entity_counts.csv')
    train_tokens_df['tag'].value_counts().to_csv('train_token_tag_counts.csv')

    print('Exploration complete. Summaries saved to CSV files.')
