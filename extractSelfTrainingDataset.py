from transformers import WhisperProcessor, WhisperForConditionalGeneration

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
tokenizer = processor.tokenizer
import fasttext
from huggingface_hub import hf_hub_download

#get language from commandline arguemnt
import sys
from tqdm import tqdm
import torchaudio
from comet import download_model, load_from_checkpoint
import torch
model_path = download_model("NataliaKhaidanova/wmt21-comet-qe-mqm")
qemodel = load_from_checkpoint(model_path)
language1 = sys.argv[1]
language2 = sys.argv[2]
maxCount = int(sys.argv[3])
model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
idmodel = fasttext.load_model(model_path)
from datasets import load_dataset
#read in lanugagePairs
path = "languagePairs.txt"
data = open(path, "r")
languagePairs = dict(eval(data.read()))
languageCodesPath = "languageCodes.txt"
data = open(languageCodesPath, "r")
languageCodes = dict(eval(data.read()))
if language1 == "afrikaans":
    myCode = "af"
else:
    myCode = languageCodes[language1]
dataset = load_dataset("mozilla-foundation/common_voice_16_1", myCode, split="train")
def detectRepetition(text):
    # if there are more than 4 spaces in the text split it
    if text.count(" ") > 4:
        text = text.split(" ")
    else:
        text = list(text)
    counts = {}
    for word in text:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    if max(counts.values())/len(text) > 0.20 and len(text) > 10:
        return True
    else:
         return False

from pathlib import Path
savePath = Path("/home/st392/fsl_groups/grp_nlp/compute/") /"mt"/"selfTrainQE"/language1  

mySavePath = savePath / language2
mySavePath.mkdir(parents=True, exist_ok=True)
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language2, task="transcribe")
count = 0
failedCount = 0
label = languagePairs[language2]
#if the label is a list
if type(label) == list:
    #append together
    label = "".join(label)
for sample in tqdm(dataset):
    audio= sample["audio"]
    audioArray = torch.from_numpy(audio["array"]).float()
    audioArray = torchaudio.transforms.Resample(audio["sampling_rate"], 16000)(audioArray)
    input_features = processor(audioArray, sample_rate=16000, return_tensors="pt").input_features

    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
    detokenizedText = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    languages = idmodel.predict(detokenizedText, k=1)
    
    if languages[0][0] in label:
        src = sample["sentence"]
        data = [{"src": src, "mt": detokenizedText}]
        qeresult = qemodel.predict(data,gpus=0)
        score = qeresult["scores"][0]
        print(f"src: {src}, mt: {detokenizedText}, qe: {score}")
        if score > 0.119 and detectRepetition(detokenizedText) == False:
            samplePath = Path(sample["path"])
            samplePath = mySavePath / samplePath.with_suffix(".txt").name
            with open(samplePath, "w") as file:
                file.write(detokenizedText)
            count += 1
    else:
        failedCount += 1
    if count >= maxCount:
        break
print(f"finished {language1} to {language2} with {count} samples, failed {failedCount} samples, failed percent {failedCount/(count+failedCount)}")
#put the pcont and failed count in npy 
counts = {}
counts["pcont"] = count
counts["failed"] = failedCount
import numpy as np
filePath = mySavePath / "counts.npy"
np.save(filePath, counts)

    
