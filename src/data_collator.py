import torch
from random import random, randint

# data collator pseudocode for intent prediction
tokens_a = ['[INT]', 'How', 'are', 'you', 'today', '?']
tokens_b = ['[INT]', 'What', 'is', 'going', 'on', '?']
intent_embeddings = {'DES': embedding1, 'QS': embedding2, 'CODE': embedding3, 'ST': embedding4}

intent_labels = []
if random() < 0.05:
    random_intent = randint(0, 3)
    tokens_b[0].embedding = intent_embeddings[list(intent_embeddings.keys())[random_intent]]
else:
    tokens_b[0].embedding = tokens_b[0].embedding

if tokens_a[0].embedding == tokens_b[0].embedding:
    is_intent_diff = False

else:
    is_intent_diff = True
intent_labels.append(torch.tensor(1 if is_intent_diff else 0))

# loss function for intent prediction
intent_score = torch.tensor # from nn.Linear(cfg.embsize, 2)
loss_fct = torch.nn.CrossEntropyLoss()
intent_loss = loss_fct(intent_score.view(-1, 2), intent_labels.view(-1))