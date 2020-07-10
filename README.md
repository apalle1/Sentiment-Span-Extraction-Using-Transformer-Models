# Sentiment-Span-Extraction-Using-Transformer-Models

## Objective

Pick out the part of the tweet (word or phrase) that reflects the labeled sentiment (positive, negative, neutral). In other words, construct a model that can look at the labeled sentiment for a given tweet and figure out what word or phrase best supports it.

```
Examples:-

Tweet: "My ridiculous dog is amazing." 
Labeled sentiment: positive
Predict: "amazing"

Tweet: "Sick. With a flu like thing." 
Labeled sentiment: negative
Predict: "Sick."
```

## Models 

Albert Large V2 - https://huggingface.co/albert-large-v2

Bert Base Uncased - https://huggingface.co/bert-base-uncased

Bert Large Uncased WWM Finetuned Squad - https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad

Distil Roberta Base - https://huggingface.co/distilroberta-base

Roberta Base Squad2 - https://huggingface.co/deepset/roberta-base-squad2

Roberta large Squad2 - https://huggingface.co/a-ware/roberta-large-squadv2

## Hyperparameters

## Preprocessing 

## Model Architecture

## Loss
Crossentropy, Label smoothing, Smotthing the targets (winner solution)

## Training Schedule

## Optimizer
