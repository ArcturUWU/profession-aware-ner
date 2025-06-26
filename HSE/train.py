import re
import os
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from natasha import Segmenter, Doc
from seqeval.metrics import precision_score, recall_score, f1_score

# ================== Конфигурация ==================
DATA_DIR = 'nerel_dataset/data/NEREL-v1.1'
MODEL_NAME = 'DeepPavlov/rubert-base-cased'
OUTPUT_DIR = './ner_profession_model'

MAX_LENGTH = 512
STRIDE = 128
PAD_TO_MULTIPLE_OF = 8

EPOCHS = 20
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 50
EARLY_STOPPING_PATIENCE = 3

B_PENALTY = 2.5  # Штраф для ложных B-PROFESSION
I_PENALTY = 1.5  # Штраф для ложных I-PROFESSION
# ================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrecisionOrientedTrainer(Trainer):
    """
    Trainer subclass with custom loss to prioritize high precision on PROFESSION.
    Applies heavier penalty for false positives on B-PROFESSION vs I-PROFESSION.
    """
    def __init__(
        self,
        b_penalty: float = B_PENALTY,
        i_penalty: float = I_PENALTY,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        cfg = self.model.config
        self.b_id = cfg.label2id.get('B-PROFESSION', -1)
        self.i_id = cfg.label2id.get('I-PROFESSION', -1)
        self.b_penalty = b_penalty
        self.i_penalty = i_penalty
        if self.b_id < 0 or self.i_id < 0:
            logger.warning("PROFESSION labels not found in label2id mapping.")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        num_labels = logits.size(-1)

        ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
            logits.view(-1, num_labels), labels.view(-1)
        )

        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
        b_prob = probs[..., self.b_id]
        i_prob = probs[..., self.i_id]

        neg_mask = (
            (labels != self.b_id) &
            (labels != self.i_id) &
            (labels != -100)
        )
        penalty = torch.tensor(0.0, device=ce_loss.device)
        if neg_mask.sum() > 0:
            penalty = (
                self.b_penalty * b_prob[neg_mask] +
                self.i_penalty * i_prob[neg_mask]
            ).mean()

        loss = ce_loss + penalty
        return (loss, outputs) if return_outputs else loss

    def plot_training(self):
        logs = self.state.log_history
        epochs, val_precision = [], []
        steps, train_loss = [], []
        for log in logs:
            if 'step' in log and 'loss' in log and 'eval_loss' not in log:
                steps.append(log['step']); train_loss.append(log['loss'])
            if 'eval_precision' in log:
                epochs.append(log['epoch']); val_precision.append(log['eval_precision'])

        plt.figure(); plt.plot(epochs, val_precision)
        plt.xlabel('Epoch'); plt.ylabel('Validation Precision')
        plt.title('Precision on PROFESSION'); plt.grid(True); plt.show()

        plt.figure(); plt.plot(steps, train_loss)
        plt.xlabel('Step'); plt.ylabel('Training Loss')
        plt.title('Loss Curve'); plt.grid(True); plt.show()


def load_brat_data(directory: str):
    segmenter = Segmenter()
    all_tokens, all_tags = [], []
    ann_pattern = re.compile(r"^T\d+\t(\w+)\s+(\d+)\s+(\d+)")

    for fname in os.listdir(directory):
        if not fname.endswith('.txt'): continue
        base = os.path.splitext(fname)[0]
        txt_path = os.path.join(directory, base + '.txt')
        ann_path = os.path.join(directory, base + '.ann')

        text = open(txt_path, encoding='utf-8').read()
        entities = []
        if os.path.exists(ann_path):
            for line in open(ann_path, encoding='utf-8'):
                m = ann_pattern.match(line.strip())
                if not m: continue
                label, start, end = m.group(1), int(m.group(2)), int(m.group(3))
                entities.append({'start': start, 'end': end, 'label': label})

        doc = Doc(text); doc.segment(segmenter)
        tokens = doc.tokens; tags = ['O'] * len(tokens)
        for ent in sorted(entities, key=lambda e: e['start']):
            started = False
            for i, tok in enumerate(tokens):
                if tok.start < ent['end'] and tok.stop > ent['start']:
                    prefix = 'B' if not started else 'I'
                    tags[i] = f"{prefix}-{ent['label']}"; started = True

        all_tokens.append([t.text for t in tokens])
        all_tags.append(tags)

    return {'tokens': all_tokens, 'ner_tags_str': all_tags}


def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokens = tokenizer(
        examples['tokens'], is_split_into_words=True,
        truncation=True, padding='max_length',
        max_length=MAX_LENGTH, stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        pad_to_multiple_of=PAD_TO_MULTIPLE_OF
    )
    sample_map = tokens.pop('overflow_to_sample_mapping')
    offsets = tokens.pop('offset_mapping')
    labels = []
    for i, offset in enumerate(offsets):
        word_ids = tokens.word_ids(batch_index=i)
        src_idx = sample_map[i]; prev = None
        lab_ids = []
        for wid in word_ids:
            if wid is None: lab_ids.append(-100)
            elif wid != prev: lab_ids.append(label2id[examples['ner_tags_str'][src_idx][wid]])
            else: lab_ids.append(label2id[examples['ner_tags_str'][src_idx][wid]])
            prev = wid
        labels.append(lab_ids)
    tokens['labels'] = labels; return tokens

# ======== Основной скрипт ========
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Загрузка данных
    train_data = load_brat_data(os.path.join(DATA_DIR, 'train'))
    dev_data = load_brat_data(os.path.join(DATA_DIR, 'dev'))

    train_ds = Dataset.from_dict(train_data)
    dev_ds = Dataset.from_dict(dev_data)
    all_tags = train_data['ner_tags_str'] + dev_data['ner_tags_str']
    unique_tags = sorted({tag for seq in all_tags for tag in seq})
    label2id = {t: i for i, t in enumerate(unique_tags)}
    id2label = {i: t for t, i in label2id.items()}

    train_ds = train_ds.map(lambda ex: {'ner_tags': [label2id[t] for t in ex['ner_tags_str']]}, batched=False)
    dev_ds = dev_ds.map(lambda ex: {'ner_tags': [label2id[t] for t in ex['ner_tags_str']]}, batched=False)

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    train_tk = train_ds.map(lambda ex: tokenize_and_align_labels(ex, tokenizer, label2id), batched=True, remove_columns=train_ds.column_names)
    dev_tk   = dev_ds.map(lambda ex: tokenize_and_align_labels(ex, tokenizer, label2id), batched=True, remove_columns=dev_ds.column_names)

    model = BertForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(unique_tags), id2label=id2label, label2id=label2id
    ).to(device)

    def compute_metrics(p):
        logits, labels = p; preds = np.argmax(logits, axis=-1)
        true_l, true_p = [], []
        for lab_seq, pred_seq in zip(labels, preds):
            seq_l, seq_p = [], []
            for l, p in zip(lab_seq, pred_seq):
                if l == -100: continue
                seq_l.append(id2label[l]); seq_p.append(id2label[p])
            true_l.append(seq_l); true_p.append(seq_p)
        return {'precision': precision_score(true_l, true_p), 'recall': recall_score(true_l, true_p), 'f1': f1_score(true_l, true_p)}

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
        learning_rate=LEARNING_RATE, per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE, num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY, logging_steps=LOGGING_STEPS,
        logging_dir=os.path.join(OUTPUT_DIR, 'logs'), metric_for_best_model='precision',
        load_best_model_at_end=True, no_cuda=not torch.cuda.is_available(), fp16=torch.cuda.is_available(), dataloader_pin_memory=True
    )

    trainer = PrecisionOrientedTrainer(
        model=model, args=args,
        train_dataset=train_tk, eval_dataset=dev_tk,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer, compute_metrics=compute_metrics
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE))

    logger.info("=== Start training ===")
    trainer.train(); logger.info("=== Training completed ===")
    trainer.save_model(OUTPUT_DIR); logger.info(f"Model saved to {OUTPUT_DIR}")
    trainer.plot_training()
