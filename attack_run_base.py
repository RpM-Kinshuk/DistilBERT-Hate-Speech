import pandas as pd
import textattack
from transformers import DistilBertForSequenceClassification, AutoTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import Dataset
from textattack.attack_recipes import TextFoolerJin2019
from textattack import AttackArgs, Attacker

class CustomDataset(Dataset):
    def __init__(self, csv_file, shuffle=False):
        self.df = pd.read_csv(csv_file)
        self.texts = self.df['text'].tolist()
        self.labels = self.df['label'].tolist()
        self.shuffle = shuffle

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (self.texts[idx], self.labels[idx])

    @property
    def shuffled(self):
        return self.shuffle

# Step 1: Load your fine-tuned model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("/rscratch/tpang/kinshuk/RpMKin/distilbert/model")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Step 2: Wrap your model with TextAttack's wrapper 
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# Step 3: Choose an attack
attack = TextFoolerJin2019.build(model_wrapper)

# Step 4: Load your dataset
dataset = CustomDataset('/rscratch/tpang/kinshuk/RpMKin/distilbert/dataset/english/data.csv')

# Step 5: Run the attack
attack_args = AttackArgs(num_examples=int(0.01*len(dataset)))
attacker = Attacker(attack, dataset, attack_args)
res = attacker.attack_dataset()
print(res.__dict__)
