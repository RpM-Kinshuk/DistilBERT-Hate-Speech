import textattack
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import TextFoolerJin2019

# Step 1: Load your fine-tuned model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("path_to_your_model")
tokenizer = AutoTokenizer.from_pretrained("path_to_your_model")

# Step 2: Wrap your model with TextAttack's wrapper (NEED TO CHANGE AS PER REQ)
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# Step 3: Choose an attack recipe (NEED to change)

attack = TextFoolerJin2019.build(model_wrapper)

# Step 4: Run the attack
#example dataset used
dataset = HuggingFaceDataset("ag_news", None, "test")

# Attack the first n samples from the dataset
for i in range(n):
    result = attack.attack_dataset(dataset, indices=[i])
    print(result)
