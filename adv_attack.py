import pandas as pd
from transformers import DistilBertForSequenceClassification, AutoTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset
from textattack.attack_recipes import (
    TextFoolerJin2019,
    HotFlipEbrahimi2017,
    BERTAttackLi2020,
    DeepWordBugGao2018,
    InputReductionFeng2018,
    BAEGarg2019,
)
from textattack import AttackArgs, Attacker
import os

attack_dict = {
    "textfooler": TextFoolerJin2019,
    "hotflip": HotFlipEbrahimi2017,
    "bert": BERTAttackLi2020,
    "deepwordbug": DeepWordBugGao2018,
    "inputreduction": InputReductionFeng2018,
}

# Step 1: Load fine-tuned model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained(
    "/rscratch/tpang/kinshuk/RpMKin/distilbert/model"
)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Step 2: Wrap model with TextAttack's wrapper
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# Read a CSV and geenrate a list of tuples from columns 'text' and 'labels'
df = pd.read_csv("/rscratch/tpang/kinshuk/RpMKin/distilbert/dataset/english/data.csv")
texts = df["text"].tolist()
labels = df["label"].tolist()
label_names = ["0", "1", "2"]
data = list(zip(texts, labels))

# Step 4: Load your dataset
dataset = Dataset(data, input_columns=["text"], label_names=label_names, shuffle=True)

# Step 5: Run the attack
attack_args = AttackArgs(
    num_examples=1000,
    log_to_csv="BAEGarg2019_results.csv",
    # checkpoint_interval=5,
    checkpoint_dir=os.path.join(os.getcwd(), "checkpoints"),
    disable_stdout=False,
)
attack = BAEGarg2019.build(model_wrapper)
attacker = Attacker(attack, dataset, attack_args)
res = attacker.attack_dataset()