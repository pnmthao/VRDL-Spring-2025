# NYCU Selected Topics in Visual Recognition using Deep Learning 2025 Spring HW1
# StudentID: 413540004
# StudentName: Phan Nguyen Minh Thao (潘阮明草)
from timm.loss import LabelSmoothingCrossEntropy
from transformers import TrainerCallback, PretrainedConfig, PreTrainedModel
from copy import deepcopy
import timm
from datasets import load_dataset
import random
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import torch
from evaluate import load
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision import datasets
from torch.utils.data import DataLoader
import os
# remove warnings
import warnings
warnings.filterwarnings("ignore")

seed = 42
random.seed(seed)

parser = argparse.ArgumentParser()

parser.add_argument('-e', "--epochs", type=int, default=25)
parser.add_argument('-l', "--lr", type=float, default=0.001)
parser.add_argument('-m', "--model_name", type=str,
                    default="timm/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288")
parser.add_argument('-b', "--batch_size", type=int, default=64)
parser.add_argument('-s', "--scheduler", type=str,
                    default='cosine', choices=['cosine', 'cosine_with_restarts', 'linear', 'constant', 'constant_with_warmup'])
parser.add_argument('-o', "--optimizer", type=str,
                    default='adagrad', choices=['adagrad', 'sgd', 'adamw_torch'])
parser.add_argument('-c', "--loss", type=str, default='ce',
                    choices=['ce', 'label_ce'])
parser.add_argument('-n', "--notes", type=str, default='')

args = parser.parse_args()
print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

dataset = load_dataset("imagefolder", data_dir="data/")
labels = dataset["train"].features["label"].names

model_config = timm.create_model(args.model_name, pretrained=False)
data_config = timm.data.resolve_model_data_config(model_config)
train_transforms = timm.data.create_transform(**data_config, is_training=True)
val_transforms = timm.data.create_transform(**data_config, is_training=False)
del model_config

print(train_transforms)
print(val_transforms)


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(
        image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


train_ds, val_ds = dataset['train'], dataset['validation']
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

try:
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        ignore_mismatched_sizes=True,
        problem_type="single_label_classification"
    )
    print(AutoModelForImageClassification)
except:
    # Create a custom configuration
    class TimmResNetConfig(PretrainedConfig):
        model_type = "timm_resnet"

        def __init__(self, num_classes=100, **kwargs):
            super().__init__(**kwargs)
            self.num_classes = num_classes

    # Instantiate config
    config = TimmResNetConfig(num_classes=100)

    class TimmResNetForImageClassification(PreTrainedModel):
        config_class = TimmResNetConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = timm.create_model(
                args.model_name, num_classes=config.num_classes, pretrained=True)

        def forward(self, pixel_values, labels=None):
            logits = self.model(pixel_values)
            return {"logits": logits}

    # Load the model
    model = TimmResNetForImageClassification(config)
    print('TimmResNetForImageClassification')

model = model.to(device)
print('#parameters:', sum(p.numel() for p in model.parameters()))

datestr = f"{datetime.now().strftime('%Y%m%d__%H%M%S')}"
filename = f"{datestr}_seed{seed}_epoch{args.epochs}_lr{args.lr}_bs{args.batch_size}_{args.optimizer}_{args.scheduler}_{args.loss}_{args.notes}"
result_dir = f"results/{args.model_name.split('/')[1]}/{args.loss}/{filename}"

os.makedirs(result_dir, exist_ok=True)
os.system(f"cp 413540004.py {result_dir}/")  # backup code
print(result_dir)

train_args = TrainingArguments(
    result_dir,
    seed=seed,
    remove_unused_columns=False,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,

    optim=args.optimizer,
    lr_scheduler_type=args.scheduler,
    learning_rate=args.lr,
    warmup_ratio=0.1,
    load_best_model_at_end=True,
    report_to="none",
    metric_for_best_model="eval_accuracy",
    push_to_hub=False,
    dataloader_num_workers=16,
    fp16=True,  # specify bf16=True instead when training on GPUs that support bf16
)

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
metric = load("accuracy")


def compute_metrics(eval_pred):
    global val_acc, val_loss
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"]
                               for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if args.loss == 'ce':
            self.custom_loss = torch.nn.CrossEntropyLoss()
            print("CrossEntropyLoss")
        elif args.loss == 'label_ce':
            self.custom_loss = LabelSmoothingCrossEntropy()
            print('LabelSmoothingCrossEntropy')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.custom_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss


class CustomCallback(TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


trainer = CustomTrainer(
    model=model,
    args=train_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
trainer.add_callback(CustomCallback(trainer))

train_results = trainer.train()
# rest is optional but nice to have
trainer.save_model(f"{result_dir}/finetuned")
trainer.log_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate()
# some nice to haves:
trainer.log_metrics("eval", metrics)
best_val_acc = metrics['eval_accuracy']

# Evalating Phase
model.eval()

test_data = datasets.ImageFolder("data/test", transform=val_transforms)
print('Classes (testlabel):', test_data.classes)
test_loader = DataLoader(
    test_data, batch_size=args.batch_size, shuffle=False, num_workers=6)
test_imgs = [filepath.split('/')[-1].split('.')[0]
             for filepath, _ in test_data.imgs]

predictions = []
for inputs, _ in test_loader:
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(inputs).logits
        _, preds = torch.max(outputs, 1)  # used for accuracy
        predictions += list(preds.detach().cpu().numpy())

df = pd.DataFrame(test_imgs, columns=['image_name'])
df['pred_label'] = predictions
df = df.sort_values(by=['image_name'])

filename = f"{filename}_val{best_val_acc:.6f}"
df.to_csv(f"{result_dir}/{filename}.csv", index=False)
df.to_csv(f"{result_dir}/prediction.csv", index=False)
os.system(f"zip -j {result_dir}/{filename}.zip {result_dir}/prediction.csv")
print(f"saved successfully! {result_dir}/{filename}.csv")
