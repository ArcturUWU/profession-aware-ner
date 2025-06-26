import os
import sys
import logging
import pandas as pd
import spacy
from spacy.matcher import Matcher
from transformers import BertTokenizerFast, BertForTokenClassification
import torch
from typing import List, Tuple

# ============ Конфигурация ============
MODEL_DIR = './ner_profession_model'
SPACY_MODEL = 'ru_core_news_sm'
INPUT_FILE = 'test_texts.xlsx'
OUTPUT_FILE = 'test_texts_inference.json'
MAX_LENGTH = 512
STRIDE = 128
# ======================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_spacy_model():
    """
    Загрузка spaCy модели и инициализация Matcher для расширенных шаблонных NP.
    Совмещает несколько шаблонов для лучшего покрытия:
    - ADJ* + NOUN+
    - NOUN+ + ADP + NOUN+
    - NOUN+ (compound nouns)
    """
    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        sys.exit(
            f"SpaCy model '{SPACY_MODEL}' not found. Install via:"
            f"    python -m spacy download {SPACY_MODEL}"
        )
    matcher = Matcher(nlp.vocab)
    # Основные паттерны
    patterns = [
        # прилагательные* + существительные+
        [{'POS': 'ADJ', 'OP': '*'}, {'POS': 'NOUN', 'OP': '+'}],
        # существительные+ + предлог + существительные+
        [{'POS': 'NOUN', 'OP': '+'}, {'POS': 'ADP', 'OP': '?'}, {'POS': 'NOUN', 'OP': '+'}],
        # несколько существительных подряд (compound)
        [{'POS': 'NOUN', 'OP': '{2,}'}]
    ]
    matcher.add('NP_EXTENDED', patterns)
    return nlp, matcher


def get_np_spans(text: str, nlp, matcher) -> List[Tuple[int,int,str]]:
    """
    Извлечение NP-спанов с помощью spaCy matcher по расширенным шаблонам.
    """
    spans = set()
    doc = nlp(text)
    # Применяем все указанные шаблоны
    for match_id, start, end in matcher(doc):
        span = doc[start:end]
        spans.add((span.start_char, span.end_char))
    # Фильтрация: минимум 2 токена, не все стоп-слова
    result = []
    for s, e in sorted(spans):
        segment = text[s:e].strip()
        tokens = nlp(segment)
        if len(tokens) < 2 or all(tok.is_stop for tok in tokens):
            continue
        result.append((s, e, 'NP'))
    return result


def get_ner_spans(text: str, tokenizer, model, id2label) -> List[Tuple[int,int,str]]:
    """
    Извлечение NER-спанов посредством скользящего окна.
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True,
        padding=False
    )
    spans = []
    for i in range(len(enc['input_ids'])):
        ids = torch.tensor([enc['input_ids'][i]])
        mask = torch.tensor([enc['attention_mask'][i]])
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits[0]
        preds = logits.argmax(dim=-1).tolist()
        for idx, lab in enumerate(preds):
            if mask[0, idx] == 0:
                continue
            label = id2label[lab]
            if label == 'O':
                continue
            s, e = enc['offset_mapping'][i][idx]
            if s == e:
                continue
            spans.append((s, e, label))
    # Объединение смежных токенов одной метки
    merged = []
    for s, e, l in sorted(spans, key=lambda x: (x[0], x[1])):
        if merged and l == merged[-1][2] and s <= merged[-1][1]:
            ps, pe, pl = merged[-1]
            merged[-1] = (ps, max(pe, e), pl)
        else:
            merged.append((s, e, l))
    return merged


def merge_spans(np_spans, ner_spans) -> List[Tuple[int,int,str]]:
    """
    Объединение NP и NER спанов:
    - Сохраняем все NER-спаны
    - NP-спаны добавляем, только если они не пересекаются с любым NER
    """
    merged = list(ner_spans)
    for s0, e0, l0 in np_spans:
        if not any(max(s0, s1) < min(e0, e1) for s1, e1, _ in ner_spans):
            merged.append((s0, e0, l0))
    return sorted(merged, key=lambda x: x[0])


def main():
    device = torch.device('cpu')
    logger.info(f'Using device: {device}')

    # Загрузка моделей
    nlp, matcher = load_spacy_model()

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    model = BertForTokenClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    id2label = model.config.id2label

    # Чтение входных данных
    try:
        df = pd.read_excel(INPUT_FILE, header=None, names=['text'], engine='openpyxl')
    except FileNotFoundError:
        sys.exit(f"Input file '{INPUT_FILE}' not found.")

    results = []
    for text in df['text']:
        np_sp = get_np_spans(text, nlp, matcher)
        ner_sp = get_ner_spans(text, tokenizer, model, id2label)
        merged = merge_spans(np_sp, ner_sp)
        terms = [{'text': text[s:e], 'start': s, 'end': e, 'label': lbl} for s, e, lbl in merged]
        results.append(terms)

    df['terms'] = results
    df.to_json(OUTPUT_FILE, orient='records', force_ascii=False)
    logger.info(f'Inference complete. Saved to {OUTPUT_FILE}')

if __name__ == '__main__':
    main()
