import torch
import argparse
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import DatasetDict, Audio, load_dataset, concatenate_datasets
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
import datasets
from trainer import PolyTrainer
#######################     Arguments for running whisper training        #########################

# python3 whisper.py --model_name=openai/whisper-medium.en --language=english --train_strategy=steps --learning_rate=2e-4 --num_steps=4500
parser = argparse.ArgumentParser(description='Fine-tuning script for Whisper Models of various sizes.')
parser.add_argument(
    '--model_name', 
    type=str, 
    required=False, 
    default='openai/whisper-small', 
    help='Huggingface model name to fine-tune. Eg: openai/whisper-small'
)
parser.add_argument(
    '--language', 
    type=str, 
    required=False, 
    default='Hindi', 
    help='Language the model is being adapted to in Camel case.'
)
parser.add_argument(
    '--sampling_rate', 
    type=int, 
    required=False, 
    default=16000, 
    help='Sampling rate of audios.'
)
parser.add_argument(
    '--num_proc', 
    type=int, 
    required=False, 
    default=1, 
    help='Number of parallel jobs to run. Helps parallelize the dataset prep stage.'
)
parser.add_argument(
    '--train_strategy', 
    type=str, 
    required=False, 
    default='steps', 
    help='Training strategy. Choose between steps and epoch.'
)
parser.add_argument(
    '--learning_rate', 
    type=float, 
    required=False, 
    default=1.75e-5, 
    help='Learning rate for the fine-tuning process.'
)
parser.add_argument(
    '--warmup', 
    type=int, 
    required=False, 
    default=20000, 
    help='Number of warmup steps.'
)
parser.add_argument(
    '--train_batchsize', 
    type=int, 
    required=False, 
    default=8, 
    help='Batch size during the training phase.'
)
parser.add_argument(
    '--eval_batchsize', 
    type=int, 
    required=False, 
    default=4, 
    help='Batch size during the evaluation phase.'
)
parser.add_argument(
    '--num_epochs', 
    type=int, 
    required=False, 
    default=20, 
    help='Number of epochs to train for.'
)
parser.add_argument(
    '--num_steps', 
    type=int, 
    required=False, 
    default=100000, 
    help='Number of steps to train for.'
)
parser.add_argument(
    '--resume_from_ckpt', 
    type=str, 
    required=False, 
    default=None, 
    help='Path to a trained checkpoint to resume training from.'
)
parser.add_argument(
    '--output_dir', 
    type=str, 
    required=False, 
    default='output_model_dir', 
    help='Output directory for the checkpoints generated.'
)

parser.add_argument(
    '--transcription_path', 
    type=str, 
    required=False, 
    default='til-ai-24-advanced/asr.jsonl', 
    help='Transcription path'
)

parser.add_argument(
    '--audio_folder', 
    type=str, 
    required=False, 
    default='til-ai-24-advanced/audio/', 
    help='Audios folder'
)


args = parser.parse_args()

if args.train_strategy not in ['steps', 'epoch']:
    raise ValueError('The train strategy should be either steps and epoch.')


print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')
print('Args:')
print(vars(args))
print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n')

gradient_checkpointing = True
freeze_feature_encoder = False
freeze_encoder = False

do_normalize_eval = True
do_lower_case = False
do_remove_punctuation = False
normalizer = BasicTextNormalizer()


#############################       MODEL LOADING       #####################################

feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task="transcribe")
processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

if model.config.decoder_start_token_id is None:
    raise ValueError("Make sure that config.decoder_start_token_id is correctly defined")

if freeze_feature_encoder:
    model.freeze_feature_encoder()

if freeze_encoder:
    model.freeze_encoder()
    model.model.encoder.gradient_checkpointing = False


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# SpecAugment
model.config.apply_spec_augment = True
model.config.mask_time_prob = 0.05
model.config.mask_feature_prob = 0.05


if gradient_checkpointing:
    model.config.use_cache = False


############################        DATASET LOADING AND PREP        ##########################
from datasets import load_dataset,Audio
from audiomentations import Compose, AddGaussianNoise, Gain, PitchShift, TimeStretch, Shift,OneOf



def mapper(sample):
    label = sample["audio"]
    sample["audio"] = args.audio_folder + label
    return sample



def load_all_datasets(train_val):    
    data_files = { "train": args.transcription_path }
    ds = load_dataset("json", data_files=data_files, split="train")
    
    split = ds.train_test_split(test_size=0.1).remove_columns(['key'])

    split = split.map(mapper, batch_size=8).cast_column("audio", Audio(sampling_rate=16000))
    return split[train_val]

augment = Compose([
    OneOf([
       TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5 ),
    # RoomSimulator(p=0.3),
    ], p = 0.7),
    PitchShift(min_semitones=-1, max_semitones=1, p=0.8),
    # OneOf([
    AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=0.85),
    # ], p=0.7),
    Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.4),
    ])


def augmented_speech(batch, augment):
    samples = batch["audio"]
    batch["audio"]['array'] = augment(samples=samples['array'], sample_rate=16000)
    return batch






def prepare_dataset(batch, augmentation = False):
    # load and (possibly) resample audio data to 16kHz
    audio = batch["audio"]
    if augmentation:
        audio['array'] = augment(samples=audio['array'], sample_rate=16000)
        
    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    
    # optional pre-processing steps
    transcription = batch["transcript"]
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()
    
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch

max_label_length = model.config.max_length
min_input_length = 0.0
max_input_length = 30.0
def is_in_length_range(length, labels):
    return min_input_length < length < max_input_length and 0 < len(labels) < max_label_length


print('DATASET PREPARATION IN PROGRESS...')


raw_dataset = DatasetDict()
train_aug = load_all_datasets('train').train_test_split(test_size = 0.5)['train']
train_aug = train_aug.map(lambda batch: augmented_speech(batch, augment), num_proc=1)

# concate with trainset
raw_dataset["train"] = datasets.concatenate_datasets([load_all_datasets('train'), train_aug])

raw_dataset["test"] = load_all_datasets('test')

raw_dataset = raw_dataset.map(prepare_dataset, num_proc=args.num_proc)

raw_dataset = raw_dataset.filter(
    is_in_length_range,
    input_columns=["input_length", "labels"],
    num_proc=args.num_proc,
) 




###############################     DATA COLLATOR AND METRIC DEFINITION     ########################

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
print('DATASET PREPARATION COMPLETED')


metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if do_normalize_eval:
        pred_str = [normalizer(pred) for pred in pred_str]
        label_str = [normalizer(label) for label in label_str]

    wer = 1 - metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


###############################     TRAINING ARGS AND TRAINING      ############################

if args.train_strategy == 'epoch':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        gradient_checkpointing=gradient_checkpointing,
        fp16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.num_epochs,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=True,
        optim="adamw_bnb_8bit",
        resume_from_checkpoint=args.resume_from_ckpt,
    )

elif args.train_strategy == 'steps':
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batchsize,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup,
        gradient_checkpointing=gradient_checkpointing,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        max_steps=args.num_steps,
        save_total_limit=10,
        per_device_eval_batch_size=args.eval_batchsize,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        optim="adamw_bnb_8bit",
        resume_from_checkpoint=args.resume_from_ckpt,
    )
    
    
    


trainer = PolyTrainer(
    args=training_args,
    model=model,
    train_dataset=raw_dataset["train"],
    eval_dataset=raw_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)


trainer.train()
print('DONE TRAINING')