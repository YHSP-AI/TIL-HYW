import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer

def get_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=5e-6, power=1.2, last_epoch=-1
):

    lr_init = optimizer.defaults["lr"]
    assert lr_init > lr_end, f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})"

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class PolyTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def create_scheduler(self, num_training_steps: int):
        self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(self.optimizer, 
                                                                      num_warmup_steps=self.args.warmup_steps,
                                                                      num_training_steps=num_training_steps)
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        self.create_scheduler(num_training_steps)