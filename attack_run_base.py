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

# Step 4: Load your dataset
df = pd.read_csv('dataset/data.csv')
# Assuming your CSV has columns 'text' for the input text and 'label' for the labels
texts = df['text'].tolist()
labels = df['label'].tolist()

# Step 5: Run the attack
for text, label in zip(texts, labels):
    result = attack.attack(text, label)
    print(result)
