from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset
from pathlib import Path
# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
feature_extractor = processor.feature_extractor
tokenizer = processor.tokenizer
import torchaudio
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from tqdm import tqdm
import torch


class SingleDataset(torch.utils.data.Dataset):
    def __init__(self,translatePrompt=False,test=False):
        datasetPath = Path("/home/st392/fsl_groups/grp_nlp/compute/mt/datasets/iwslt2023_mr-hi")
        if test:
            datasetPath = datasetPath / "test"
        else:
            datasetPath = datasetPath / "train"
        self.translatePrompt = translatePrompt
        textFilePath = datasetPath/"txt"/f"{datasetPath.stem}.hi"
        self.startTrascriptToken = 50258
        wavFiles = list(datasetPath.rglob("*.wav"))
        wavFiles = {int(file.stem.split("_")[-1]): file for file in wavFiles}
        #sort the files based on key value low to high int
        wavFiles = dict(sorted(wavFiles.items(), key=lambda item: item[0]))
        #values as list 
        wavFiles = list(wavFiles.values())
        self.test = test
        self.textData, self.wavData = None,None
        self.loadDataset(textFilePath, wavFiles)
        
    def loadDataset(self, textFile, wavFiles):
        self.wavData = {}
        self.textData = {}
        lines = textFile.read_text().split("\n")
        language1 = "marathi"
        language2 = "hindi"
        for i,(line,audioFile) in enumerate(tqdm(zip(lines, wavFiles))):
            tokenizedText = tokenizer(line, return_tensors="pt")
            if self.translatePrompt and language1 != language2:
                #translation prompt for language2
                forced_decoder_ids = processor.get_decoder_prompt_ids(language=language1, task="translate")    
                #get langauge 2 transcribe prompt to get token
                forced_decoder_ids2 = processor.get_decoder_prompt_ids(language=language2, task="transcribe")
                language1Token = forced_decoder_ids[0][1]
                language2Toekn = forced_decoder_ids2[0][1]
                translateToken = forced_decoder_ids[1][1]
                transcribeToken = forced_decoder_ids2[1][1]
                noTimeStampToken = forced_decoder_ids2[2][1]
                forced_decoder_ids = [self.startTrascriptToken, language1Token, translateToken, language2Toekn,  noTimeStampToken]
            elif self.translatePrompt and language1 == language2:
                #translation prompt for language2
                forced_decoder_ids = processor.get_decoder_prompt_ids(language=language1, task="transcribe")
                forced_decoder_ids = [self.startTrascriptToken] + [id[1] for id in forced_decoder_ids]

            else:
                #transcription prompt for language2
                forced_decoder_ids = processor.get_decoder_prompt_ids(language=language2, task="transcribe")
                language2Toekn = forced_decoder_ids[0][1]
                transcribeToken = forced_decoder_ids[1][1]
                noTimeStampToken = forced_decoder_ids[2][1]
                forced_decoder_ids = [self.startTrascriptToken, language2Toekn, transcribeToken, noTimeStampToken]
            
            #remove toekns from toeknizedText before the start token 50364
            if tokenizedText["input_ids"].shape[1]+len(forced_decoder_ids) > 448:
                continue
            self.textData[i+1] = {"tokenizedText": tokenizedText, "forced_decoder_ids": forced_decoder_ids}
            audio, sample_rate = torchaudio.load(audioFile)
            if audio.shape[0] == 2:
                #change to mono
                audio = audio.mean(0).unsqueeze(0)
            audio = audio.squeeze(0)
            audio = feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
            self.wavData[i+1] = audio
        return self.textData, self.wavData
    def __len__(self):
        return len(self.textData)
    def __getitem__(self, idx):
        key = list(self.textData.keys())[idx]
        labels = self.textData[key]["tokenizedText"]["input_ids"].squeeze(0)
        input_features = self.wavData[key].squeeze(0)
        decoder_input_ids = self.textData[key]["forced_decoder_ids"]
        #append labels to decoder input ids
        if decoder_input_ids:
            labels = decoder_input_ids +labels[4:].tolist()
            labels = torch.tensor(labels, dtype=torch.long).cuda()
        if self.test:
            decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long).cuda()
            return {"input_features": input_features, "labels": labels, "prompt_ids": decoder_input_ids}
        return {"labels": labels, "input_features": input_features}

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt",padding =True)
        
        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # pad the decoder input ids to max length

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        if "prompt_ids" not in features[0]:
            return batch
        decoder_input_ids = [{"input_ids": feature["prompt_ids"]} for feature in features]
        decoder_input_ids = self.processor.tokenizer.pad(decoder_input_ids, return_tensors="pt")
        decoder_input_ids = decoder_input_ids["input_ids"].masked_fill(decoder_input_ids.attention_mask.ne(1), -100)
        batch["prompt_ids"] = decoder_input_ids
        return batch
import evaluate

bleu_metric = evaluate.load("bleu")
chrf_metric = evaluate.load("chrf")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode the predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute BLEU score. Note that BLEU expects a list of references for each prediction,
    # so we need to adjust the format. Each reference should be a list of strings.
    references = [[reference] for reference in label_str]  # Adjust for BLEU format
    bleu_score = bleu_metric.compute(predictions=pred_str, references=references)

    # Compute chrF2 score. Note that chrF expects the same format as BLEU for predictions and references.
    chrf_score = chrf_metric.compute(predictions=pred_str, references=references, beta=2)  # chrF with beta=2 emphasizes recall

    return {"bleu": bleu_score["bleu"], "chrf2": chrf_score["score"]}

if __name__ == "__main__":
    #get lr arg
    import sys
    lr = sys.argv[1]
    translatePrompt = sys.argv[2]
    translatePrompt = True if translatePrompt == "True" else False
    dataset = SingleDataset(translatePrompt=translatePrompt, )
    testDataset = SingleDataset(translatePrompt=translatePrompt, test=True)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    model.generation_config.language = "hindi"
    model.generation_config.task = "transcribe"
    baseOutputDir = Path("/home/st392/fsl_groups/grp_nlp/compute/mt/resultsSingle/")
    myOutputDir = baseOutputDir / f"lr_{lr}_translatePrompt_{translatePrompt}"
    myOutputDir.mkdir(exist_ok=True, parents=True)
    model.generation_config.forced_decoder_ids = None
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    # # freeze the first n layers of the decoder
    # freeze_layers = 0
    # for i, param in enumerate(model.model.decoder.layers.parameters()):
    #     if i >= freeze_layers:
    #         param.requires_grad = False
    training_args = Seq2SeqTrainingArguments(
        output_dir=myOutputDir,  # change to a repo name of your choice
        per_device_train_batch_size=32,
        gradient_accumulation_steps=16,  # increase by 2x for every 2x decrease in batch size
        learning_rate=float(lr),
        warmup_steps=0,
        num_train_epochs=20,
        per_device_eval_batch_size=64,
        predict_with_generate=True,
        generation_max_length=225,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_strategy="steps",        
        logging_steps=1,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        push_to_hub=False,
        label_names=["labels", "prompt_ids", "input_features"]

    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset,
        eval_dataset=testDataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()