from datasets import load_dataset

#  DEBERTA: https://huggingface.co/blog/Valerii-Knowledgator/multi-label-classification

dataset = load_dataset('knowledgator/events_classification_biotech')
for split_name, split in dataset.items():
    print(f"Split: {split_name}, Num Examples: {len(split)}")

classes = [class_ for class_ in dataset['train'].features['label 1'].names if class_]
class2id = {class_: id for id, class_ in enumerate(classes)}
id2class = {id: class_ for class_, id in class2id.items()}

from transformers import AutoTokenizer

model_path = 'microsoft/deberta-v3-small'

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Vocabulary size
vocab_size = tokenizer.vocab_size
print("Vocabulary size:", vocab_size)

# Embedding dimension
embedding_dim = tokenizer.model_max_length
print("Embedding dimension:", embedding_dim)

# Special tokens
bos_token = tokenizer.bos_token
eos_token = tokenizer.eos_token
pad_token = tokenizer.pad_token
print("BOS token:", bos_token)
print("EOS token:", eos_token)
print("PAD token:", pad_token)

# Tokenizer type
tokenizer_type = type(tokenizer).__name__
print("Tokenizer type:", tokenizer_type)


def preprocess_function(example):
    text = f"{example['title']}.\n{example['content']}"
    all_labels = example['all_labels']
    labels = [0. for i in range(len(classes))]
    for label in all_labels:
        label_id = class2id[label]
        labels[label_id] = 1.

    example = tokenizer(text, truncation=True)
    example['labels'] = labels
    return example


tokenized_dataset = dataset.map(preprocess_function)

for split_name, split in tokenized_dataset.items():
    print(f"Split: {split_name}, Num Examples: {len(split)}")

# Create an iterator for the items of tokenized_dataset
iterator = iter(tokenized_dataset.items())

# Get the first item from the iterator
first_item = next(iterator)

# Print the first item
print("First item of tokenized_dataset.items():", first_item)

# Get the first example from the dataset
first_example = tokenized_dataset['train'][0]

# Extract the tokenized input representations
input_ids = first_example['input_ids']
token_type_ids = first_example['token_type_ids']
attention_mask = first_example['attention_mask']

# Print the tokenized input representations
print("Input IDs:", input_ids)
print("Token Type IDs:", token_type_ids)
print("Attention Mask:", attention_mask)

# Initialize lists to store feature lengths and label lengths
feature_lengths = []
label_lengths = []

# Iterate over the tokenized dataset and calculate the lengths
for example in tokenized_dataset['train']:
    # Get the length of features
    feature_length = len(example['input_ids'])  # Assuming 'input_ids' is used for features
    feature_lengths.append(feature_length)

    # Get the length of labels
    label_length = len(example['labels'])  # Assuming 'labels' is used for labels
    label_lengths.append(label_length)

# Calculate the average length of features and labels
avg_feature_length = sum(feature_lengths) / len(feature_lengths)
avg_label_length = sum(label_lengths) / len(label_lengths)

# Print the results
print("Average feature length:", avg_feature_length)
print("Average label length:", avg_label_length)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

import evaluate
import numpy as np

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(

    model_path, num_labels=len(classes),
    id2label=id2class, label2id=class2id,
    problem_type="multi_label_classification")

training_args = TrainingArguments(

    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset["train"],
                  eval_dataset=tokenized_dataset["test"],
                  tokenizer=tokenizer,
                  data_collator=data_collator,
                  compute_metrics=compute_metrics)

trainer.train()

