import torch
import torch.nn.functional as F

class BeamSearchGenerator:
    def __init__(self, model, config):
        self.model = model
        self.beam_size = config.beam_size
        self.max_length = config.max_seq_length
        self.length_penalty = config.length_penalty
        
    def generate(self, src, src_mask=None):
        batch_size = src.size(0)
        
        # Encode source sequence
        encoder_output = self.model.encode(src, src_mask)
        
        # Initialize beam
        beam = [(torch.zeros((batch_size, 1), device=src.device, dtype=torch.long),
                0.0)]
        
        for _ in range(self.max_length - 1):
            candidates = []
            
            for sequence, score in beam:
                if sequence[0, -1].item() == self.model.eos_token_id:
                    candidates.append((sequence, score))
                    continue
                    
                # Get model predictions
                output = self.model.decode(sequence, encoder_output, src_mask)
                logits = output[:, -1, :]
                probs = F.log_softmax(logits, dim=-1)
                
                # Get top k candidates
                values, indices = probs.topk(self.beam_size)
                
                for i in range(self.beam_size):
                    new_sequence = torch.cat([sequence, indices[:, i:i+1]], dim=-1)
                    new_score = score + values[:, i].item()
                    candidates.append((new_sequence, new_score))
            
            # Select top beam_size candidates
            candidates.sort(key=lambda x: x[1] / (len(x[0])**self.length_penalty), reverse=True)
            beam = candidates[:self.beam_size]
            
            # Early stopping if all beams ended with EOS
            if all(seq[0, -1].item() == self.model.eos_token_id for seq, _ in beam):
                break
                
        return beam[0][0]  # Return sequence with highest score 