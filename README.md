To create the self-training data run extractSelfTrainingDataset.py <source language> <target language>
To preprocess the self-training data run createDataset.py
To train you will have to add the following lines of code to your transformers library: 

![image](https://github.com/TorriePhD/WhisperMT/assets/50847306/a2407e0b-2266-492a-a264-9cfe13ad497e)


![image](https://github.com/TorriePhD/WhisperMT/assets/50847306/91fe605e-0a96-40e2-9f22-a372a47e6e0b)

Download the IWSLT 2024 train dataset by cloning this reporsitory: https://github.com/panlingua/iwslt2023_mr-hi
After those adjustments are made, you can run train.py <lr> <promping technique (True for translate prompt, False for transcribe prompt)>

If you desire to run the training on the IWSLT training set rather than the self-training data, run traingSingle.py <lr> <promping technique (True for translate prompt, False for transcribe prompt)>
