import torch
from model.transformer import DecoderOnlyTransformer
from config import TransformerConfig
from utils.trainer import GPTTrainer
from torch.utils.data import DataLoader, Dataset

# Initialize model
config = TransformerConfig()
model = DecoderOnlyTransformer(config)

# Initialize trainer
trainer = GPTTrainer(model, config)

# Training loop (you'll need to implement the data loading part)
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = trainer.train_step(batch)
        print(f"Loss: {loss:.4f}")

# Generate text
prompt = torch.tensor([1, 2, 3])  # Your tokenized prompt
generated = trainer.generate(prompt)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encoded = tokenizer(texts, 
                               max_length=max_length,
                               padding='max_length',
                               truncation=True,
                               return_tensors='pt')
        
    def __len__(self):
        return len(self.encoded['input_ids'])
        
    def __getitem__(self, idx):
        return {
            'input_ids': self.encoded['input_ids'][idx],
            'attention_mask': self.encoded['attention_mask'][idx]
        } 