import pandas as pd
import textattack
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset
from textattack.attack_recipes import TextFoolerJin2019
from textattack.data_augmentation import Augmenter
from textattack import AttackArgs, Attacker

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.texts = self.df['text'].tolist()
        self.labels = self.df['label'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (self.texts[idx], self.labels[idx])

# Step 1: Load your fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("path_to_your_model")
tokenizer = AutoTokenizer.from_pretrained("path_to_your_model")

# Step 2: Wrap your model with TextAttack's wrapper 
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# Step 3: Choose an attack (3-4 attacks to be run)
attack = TextFoolerJin2019.build(model_wrapper)

# Step 4: Load your dataset
dataset = CustomDataset('dataset/data.csv')

# Step 5: Run the attack
attack_args = AttackArgs(num_examples=len(dataset))
attacker = Attacker(attack, dataset, attack_args)
attacker.attack_dataset()
