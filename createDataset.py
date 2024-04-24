from pathlib import Path
from datasets import load_dataset
import torchaudio
import torch

datasetLabelsPath = Path("/home/st392/code/classes/WhisperMT/mt/selfTrainQE")
codePath = "languageCodes.txt"
codes = open(codePath, "r")
languageCodesPath = "languageCodes.txt"
data = open(languageCodesPath, "r")
languageCodes = dict(eval(data.read()))
from tqdm import tqdm
for language in tqdm(list(datasetLabelsPath.iterdir())):
    print(f"Working on first {language.name}")
    if language.name in ["dutch","afrikaans",]:
        continue
    myCode = languageCodes[language.name]
    cvdataset = load_dataset("mozilla-foundation/common_voice_16_1", myCode, split="train")
    paths= cvdataset["path"]
    pathsDict = {}
    for i, path in enumerate(paths):
        path = Path(path).stem
        pathsDict[path] = i
    for language2 in tqdm(list(language.iterdir())):
        print(f"Working on second {language2.name}")
        if not (language2.name in ["hindi","marathi"] or language.name  in ["hindi","marathi"]):
            continue
        if language2.name in ["dutch","afrikaans"]:
            continue
        for file in tqdm(list(language2.iterdir())):
            savePath = language/"audio"/(file.with_suffix(".wav").name)
            savePath.parent.mkdir(parents=True, exist_ok=True)
            if savePath.exists():
                continue
            if file.suffix != ".txt":
                continue
            with open(file, 'r') as f:
                text = f.read()
            #grab audio from cvdataset
            if savePath.stem not in pathsDict:
                print(f"{savePath.name} not in pathsDict")
                continue
            index = pathsDict[savePath.stem]
            audio = cvdataset[index]["audio"]
            audioArray = torch.from_numpy(audio["array"]).float()
            audioArray = torchaudio.transforms.Resample(audio["sampling_rate"], 16000)(audioArray)
            if audioArray.ndim == 1:
                audioArray = audioArray.unsqueeze(0)
            torchaudio.save(savePath, audioArray, 16000)
