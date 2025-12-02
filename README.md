# Comp4248-Project

## Dataset
The project uses the [SciCite: Citation Intent Classification](https://github.com/allenai/scicite) dataset. 

Type: Scientific Document Processing, Sentiment Analysis, Sentence Classification

Size: 11K

Description: Given an input citation sentence (“context”), classify its sentiment / intent as one 
among {background, method, comparison} 

## Directory Structure
```
COMP4248-Project/
├── data/
│   ├── raw/
│   │   ├── train.jsonl
│   │   └── test.jsonl
│   │   └── dev.jsonl
├── model/
│   ├── BERT.py
├── utils/
│   ├── preprocessing.py
```