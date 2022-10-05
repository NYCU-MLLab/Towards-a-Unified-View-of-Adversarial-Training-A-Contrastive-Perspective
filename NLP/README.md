# Self-Supervised Adversarial Training for Contrastive Sentence Embedding

## Setup

### Environment
```
pip install -r requirements.txt
```
### Dataset Download
To download the training dataset, go to the data folder by:
```
cd ./data
```
then run the following command to download the dataset:
```
./download_wiki.sh 
```

To download the evaluation dataset, go to the SentEval data folder by:
```
cd ./NLP/SentEval/data/downstream
```
then run the following command to download the dataset:
```
./download_dataset.sh 
```



## Train and Evaluate

To train the encoder with the proposed CL and evaluate with STS dataset, first download the datasets as shown above, then run the following command:
```
./run_cl.sh
```

To train the encoder with the proposed AT and evaluate with STS dataset, run the following command:
```
./run_at.sh
```

## Checkpoints

Checkpoint of the proposed CL:
https://drive.google.com/drive/folders/10CNZjD3QsI-oE428r7grFBLktmtnYTq2?usp=sharing

Checkpoint of the self-supervised AT:
https://drive.google.com/drive/folders/13Uq3KWoR-g-Y-EPTp-6Rlo3v0TFPmfli?usp=sharing

