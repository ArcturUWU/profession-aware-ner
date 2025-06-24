import pandas as pd
import spacy
from spacy.matcher import Matcher
from transformers import BertTokenizerFast, BertForTokenClassification
import torch
from typing import List, Tuple, Dict
import sys


MODEL_NAME = "ru_core_news_sm"
try:
    nlp = spacy.load(MODEL_NAME)
except OSError:
    sys.exit(
        f"SpaCy модель '{MODEL_NAME}' не найдена. Установите её командой:\n"
        f"    python -m spacy download {MODEL_NAME}\n"
        "и повторите запуск скрипта."
    )

try:
    import benepar
    if not benepar.is_loaded('benepar_ru'):
        benepar.download('benepar_ru')
    nlp.add_pipe('benepar', config={'model': 'benepar_ru'})
    use_benepar = True
except Exception:
    use_benepar = False
  
matcher = Matcher(nlp.vocab)
matcher.add(
    "NP_PATTERN",
    [[
        {"POS": "ADJ", "OP": "*"},
        {"POS": "NOUN", "OP": "+"}
    ]]
)

# 2. Загрузка обученной NER-модели и токенизатора
model_checkpoint = './ner_profession_model_from_nerel'
try:
    tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
    model = BertForTokenClassification.from_pretrained(model_checkpoint)
    model.eval()
except Exception as e:
    sys.exit(f"Не удалось загрузить модель из {model_checkpoint}: {e}")

id2label = model.config.id2label


def get_np_spans(text: str) -> List[Tuple[int, int, str]]:
    """Гибридное извлечение NP: constituency (Benepar) + Matcher-шаблоны."""
    doc = nlp(text)
    spans_set = set()

    if use_benepar:
        for sent in doc.sents:
            for const in sent._.constituents:
                if const.label_ == 'NP':
                    spans_set.add((const.start_char, const.end_char))

    for _, start, end in matcher(doc):
        spans_set.add((doc[start:end].start_char, doc[start:end].end_char))

    spans: List[Tuple[int, int, str]] = []
    for s, e in sorted(spans_set):
        span_text = text[s:e].strip()
        tokens = nlp(span_text)
        if len(tokens) < 2 or all(tok.is_stop for tok in tokens):
            continue
        spans.append((s, e, 'NP'))
    return spans


def get_ner_spans(text: str) -> List[Tuple[int, int, str]]:
    """Извлечение NER-спанов с sliding window (stride)."""
    tokens = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        padding=False
    )
    all_spans: List[Tuple[int, int, str]] = []
    for i in range(len(tokens['input_ids'])):
        input_ids = tokens['input_ids'][i]
        offset_mapping = tokens['offset_mapping'][i]
        attention_mask = tokens['attention_mask'][i]
        input_tensor = torch.tensor([input_ids])
        mask_tensor = torch.tensor([attention_mask])
        with torch.no_grad():
            outputs = model(input_ids=input_tensor, attention_mask=mask_tensor)
        logits = outputs.logits[0]
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        for idx, pred_id in enumerate(pred_ids):
            if attention_mask[idx] == 0:
                continue
            label = id2label[pred_id]
            if label == 'O':
                continue
            start_char, end_char = offset_mapping[idx]
            if start_char == end_char:
                continue
            all_spans.append((start_char, end_char, label))
    merged: List[Tuple[int, int, str]] = []
    all_spans.sort(key=lambda x: (x[0], x[1]))
    for start, end, label in all_spans:
        if merged and start <= merged[-1][1] and merged[-1][2] == label:
            last_start, last_end, _ = merged[-1]
            merged[-1] = (last_start, max(last_end, end), label)
        else:
            merged.append((start, end, label))
    return merged


def merge_spans(np_spans: List[Tuple[int,int,str]], ner_spans: List[Tuple[int,int,str]]) -> List[Tuple[int,int,str]]:
    """
    Объединение spans:
    - Во всех случаях NER-спаны приоритетны (удаляют NP при пересечении)
    - NP-спаны добавляются только если нет пересечения с любым NER
    """
    merged = list(ner_spans)
    for s0, e0, label in np_spans:
        if not any(not (e0 <= s1 or s0 >= e1) for s1, e1, _ in ner_spans):
            merged.append((s0, e0, label))
    merged.sort(key=lambda x: x[0])
    return merged

if __name__ == '__main__':
    try:
        df = pd.read_excel('test_texts.xlsx', header=None, names=['text'], engine='openpyxl')
    except FileNotFoundError:
        sys.exit("Файл 'test_texts.xlsx' не найден в рабочей директории.")

    results = []
    for text in df['text']:
        np_spans = get_np_spans(text)
        ner_spans = get_ner_spans(text)
        merged = merge_spans(np_spans, ner_spans)
        terms = [
            {'text': text[s:e], 'start': s, 'end': e, 'label': lbl}
            for s, e, lbl in merged
        ]
        results.append(terms)

    df['terms'] = results
    df.to_json('test_texts_inference.json', orient='records', force_ascii=False)
    print("Inference complete. Results saved to test_texts_inference.json")
