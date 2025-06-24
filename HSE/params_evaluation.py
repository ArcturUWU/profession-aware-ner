import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification, EarlyStoppingCallback
from datasets import Dataset
import numpy as np
import os
from natasha import Segmenter, Doc
from seqeval.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import optuna

class PrecisionOrientedTrainer(Trainer):
    def __init__(self, penalty_weight=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.model.config
        self.b_id = cfg.label2id.get('B-PROFESSION', -1)
        self.i_id = cfg.label2id.get('I-PROFESSION', -1)
        self.penalty_weight = penalty_weight
        if self.b_id < 0 or self.i_id < 0:
            print("Внимание: метки PROFESSION не найдены в label2id.")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        num_labels = logits.size(-1)

        ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')(
            logits.view(-1, num_labels), labels.view(-1)
        )

        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
        b_probs = probs[..., self.b_id]
        i_probs = probs[..., self.i_id]
        profession_prob = (b_probs + i_probs) / 2

        neg_mask = (labels != self.b_id) & (labels != self.i_id) & (labels != -100)
        penalty = profession_prob[neg_mask].mean() if neg_mask.sum() > 0 else torch.tensor(0.0, device=ce_loss.device)

        loss = ce_loss + self.penalty_weight * penalty
        return (loss, outputs) if return_outputs else loss

    def plot_training(self):
        logs = self.state.log_history
        train_loss_steps = []
        val_precision = []
        val_epochs = []
        for log in logs:
            if 'step' in log and 'loss' in log and 'eval_loss' not in log:
                train_loss_steps.append(log['loss'])
            if 'eval_precision' in log:
                val_epochs.append(log['epoch'])
                val_precision.append(log['eval_precision'])
        plt.figure()
        plt.plot(val_epochs, val_precision)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Precision')
        plt.title('Precision on PROFESSION over epochs')
        plt.grid(True)
        plt.show()
        plt.figure()
        plt.plot(range(len(train_loss_steps)), train_loss_steps)
        plt.xlabel('Training Step')
        plt.ylabel('Training Loss')
        plt.title('Training Loss over steps')
        plt.grid(True)
        plt.show()

def load_brat_data(directory: str):
    segmenter = Segmenter()
    all_tokens, all_tags = [], []

    for filename in os.listdir(directory):
        if not filename.endswith(".txt"): continue
        base = os.path.splitext(filename)[0]
        text = open(os.path.join(directory, base + '.txt'), 'r', encoding='utf-8').read()
        entities = []
        for line in open(os.path.join(directory, base + '.ann'), 'r', encoding='utf-8'):
            if line.startswith('T'):
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    _, info, _ = parts
                    info_tokens = info.split()
                    if len(info_tokens) < 3:
                        continue
                    label = info_tokens[0]
                    start = info_tokens[1].split(';')[0]
                    end = info_tokens[2].split(';')[0]
                    entities.append({'start':int(start),'end':int(end),'label':label})
        doc = Doc(text)
        doc.segment(segmenter)
        tags = ['O'] * len(doc.tokens)
        for ent in sorted(entities, key=lambda x: x['start']):
            started = False
            for i, token in enumerate(doc.tokens):
                if token.start < ent['end'] and token.stop > ent['start']:
                    tags[i] = 'B-'+ent['label'] if not started else 'I-'+ent['label']
                    started = True
        all_tokens.append([t.text for t in doc.tokens])
        all_tags.append(tags)

    return {"tokens": all_tokens, "ner_tags_str": all_tags}

def train(trial):
    base = "HSE/nerel_dataset/data/NEREL-v1.1/"
    train_data = load_brat_data(base + 'train')
    dev_data   = load_brat_data(base + 'dev')

    train_ds = Dataset.from_dict(train_data)
    dev_ds   = Dataset.from_dict(dev_data)
    all_tags = train_data['ner_tags_str'] + dev_data['ner_tags_str']
    unique = sorted({tag for seq in all_tags for tag in seq})
    label2id = {t:i for i,t in enumerate(unique)}
    id2label = {i:t for t,i in label2id.items()}

    train_ds = train_ds.map(lambda ex: { 'ner_tags': [label2id[t] for t in ex['ner_tags_str']] }, batched=False)
    dev_ds   = dev_ds.map(lambda ex: { 'ner_tags': [label2id[t] for t in ex['ner_tags_str']] }, batched=False)

    tokenizer = BertTokenizerFast.from_pretrained('DeepPavlov/rubert-base-cased')
    def tokenize_align(exs):
        tok = tokenizer(exs['tokens'], is_split_into_words=True, truncation=True,
                        padding='max_length', max_length=512, stride=128,
                        return_overflowing_tokens=True, return_offsets_mapping=True)
        sam_map = tok.pop('overflow_to_sample_mapping')
        tok.pop('offset_mapping')
        labels = []
        for i in range(len(tok['input_ids'])):
            wids = tok.word_ids(batch_index=i)
            src = sam_map[i]
            prev = None
            lab = []
            for wid in wids:
                if wid is None: lab.append(-100)
                elif wid != prev: lab.append(exs['ner_tags'][src][wid])
                else: lab.append(exs['ner_tags'][src][wid])
                prev = wid
            labels.append(lab)
        tok['labels'] = labels
        return tok

    train_tk = train_ds.map(tokenize_align, batched=True, remove_columns=train_ds.column_names)
    dev_tk   = dev_ds.map(tokenize_align, batched=True, remove_columns=dev_ds.column_names)

    model = BertForTokenClassification.from_pretrained(
        'DeepPavlov/rubert-base-cased', num_labels=len(unique), id2label=id2label, label2id=label2id
    )

    def compute_metrics(p):
        logits, labels = p
        preds = np.argmax(logits, axis=-1)
        t_labels, t_preds = [], []
        for lab_seq, pred_seq in zip(labels, preds):
            seq_l, seq_p = [], []
            for l, p in zip(lab_seq, pred_seq):
                if l == -100: continue
                seq_l.append(id2label[l]); seq_p.append(id2label[p])
            t_labels.append(seq_l); t_preds.append(seq_p)
        return { 'precision': precision_score(t_labels, t_preds),
                 'recall':    recall_score(t_labels, t_preds),
                 'f1':        f1_score(t_labels, t_preds) }

    # Define hyperparameters with Optuna
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])
    num_train_epochs = trial.suggest_categorical("num_train_epochs", [10, 15, 20])
    weight_decay = trial.suggest_categorical("weight_decay", [0.01, 0.1])
    penalty_weight = trial.suggest_categorical("penalty_weight", [1.0, 2.0, 3.0])

    args = TrainingArguments(
        output_dir='./ner_profession_model', eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
        learning_rate=learning_rate, per_device_train_batch_size=per_device_train_batch_size, per_device_eval_batch_size=8,
        num_train_epochs=num_train_epochs, weight_decay=weight_decay, logging_dir='./logs', metric_for_best_model='precision',
        load_best_model_at_end=True
    )
    args.fp16 = True

    trainer = PrecisionOrientedTrainer(
        model=model, args=args,
        train_dataset=train_tk, eval_dataset=dev_tk,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer, compute_metrics=compute_metrics,
        penalty_weight=penalty_weight
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    print("=== Start training ===")
    trainer.train()
    print("=== Training completed ===")

    metrics = trainer.evaluate()
    precision = metrics['eval_precision']

    return precision

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(train, n_trials=10) 

    print("Best trial:")
    trial = study.best_trial
    print(f"  Precision: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("Training final model with best hyperparameters...")
    best_params = trial.params
    # You can re-run train() with best_params fixed or modify the code to use them directly
