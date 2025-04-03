import torch
import torch.nn.functional as F
from torch.optim import AdamW

class GPTTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        
    def train_step(self, batch):
        self.model.train()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Shift sequences for language modeling
        labels = input_ids[:, 1:].contiguous()
        logits = self.model(input_ids[:, :-1], attention_mask[:, :-1])[0]
        
        # Calculate loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                             labels.view(-1),
                             ignore_index=self.config.pad_token_id)
        
        # Backpropagation
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
        
    @torch.no_grad()
    def generate(self, prompt, max_length=100, temperature=0.7):
        self.model.eval()
        curr_ids = prompt.unsqueeze(0)
        
        for _ in range(max_length):
            logits = self.model(curr_ids)[0][:, -1, :] / temperature
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            if next_token.item() == self.config.eos_token_id:
                break
                
        return curr_ids.squeeze(0)