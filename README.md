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

**Architecture 1**

```
class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.BertModel.from_pretrained(
            config.PRETRAINED_MODEL_PATH,
            config=conf)
        self.high_dropout = torch.nn.Dropout(config.HIGH_DROPOUT)
        self.classifier = torch.nn.Linear(config.HIDDEN_SIZE * 2, 2)

        torch.nn.init.normal_(self.classifier.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids):
        # sequence_output of N_LAST_HIDDEN + Embedding states
        # (N_LAST_HIDDEN + 1, batch_size, num_tokens, 768)
        _, _, out = self.roberta(ids, attention_mask=mask,
                                 token_type_ids=token_type_ids)

        out = torch.stack(
            tuple(out[-i - 1] for i in range(config.N_LAST_HIDDEN)), dim=0)
        out_mean = torch.mean(out, dim=0)
        out_max, _ = torch.max(out, dim=0)
        out = torch.cat((out_mean, out_max), dim=-1)

        # Multisample Dropout: https://arxiv.org/abs/1905.09788
        logits = torch.mean(torch.stack([
            self.classifier(self.high_dropout(out))
            for _ in range(5)
        ], dim=0), dim=0)

        start_logits, end_logits = logits.split(1, dim=-1)

        # (batch_size, num_tokens)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
```

**Architecture 2**

```
class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(config.ROBERTA_PATH, config=conf)
 
        self.qa_outputs1_1c = torch.nn.Conv1d(768, 128, 2)
        self.qa_outputs1_2c = torch.nn.Conv1d(128, 64, 2)
 
        self.qa_outputs2_1c = torch.nn.Conv1d(768, 128, 2)
        self.qa_outputs2_2c = torch.nn.Conv1d(128, 64, 2)
 
        self.qa_outputs1 = nn.Linear(64, 1)
        self.qa_outputs2 = nn.Linear(64, 1)
 
        self.dropout = nn.Dropout(0.1)
 
    def forward(self, ids, mask, token_type_ids):
 
        out = self.roberta(ids,
            attention_mask=mask,
            token_type_ids=token_type_ids)
 
        s_out = self.dropout(out[0])
        s_out = torch.nn.functional.pad(s_out.transpose(1,2), (1, 0))
 
        out1 = self.qa_outputs1_1c(s_out)
        out1 = self.qa_outputs1_2c(out1).transpose(1,2)
        start_logits = self.qa_outputs1(self.dropout(out1)).squeeze(-1)
 
        out2 = self.qa_outputs2_1c(s_out)
        out2 = self.qa_outputs2_2c(out2).transpose(1,2)        
        end_logits = self.qa_outputs2(self.dropout(out2)).squeeze(-1)
 
        return start_logits, end_logits
```

**Architecture 3**

```
class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(config.ROBERTA_PATH, config=conf)
 
        self.dropout = nn.Dropout(0.1)
        
        self.l0 = nn.Linear(768, 384)
        self.l1 = nn.Linear(384, 192)
        self.l2 = nn.Linear(192, 2)

        # https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
        torch.nn.init.normal_(self.l0.weight, std=0.02)
        torch.nn.init.normal_(self.l1.weight, std=0.02)
        torch.nn.init.normal_(self.l2.weight, std=0.02)
 
    def forward(self, ids, mask, token_type_ids):
        
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = self.dropout(out[-1])
 
        out = self.l0(out)
        out = self.l1(out)
        logits = self.l2(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        #print(f"Start Logit size: {start_logits.size()}")
        #print(f"End Logit size: {end_logits.size()}")
        
        return start_logits, end_logits
```

**Architecture 4**

```
class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(config.ROBERTA_PATH, config=conf)
 
        self.dropout = nn.Dropout(0.1)
        
        self.l0 = nn.Linear(768 * 2, 768)
        self.l1 = nn.Linear(768, 384)
        self.l2 = nn.Linear(384, 192)
        self.l3 = nn.Linear(192, 2)

        # https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py
        torch.nn.init.normal_(self.l0.weight, std=0.02)
        torch.nn.init.normal_(self.l1.weight, std=0.02)
        torch.nn.init.normal_(self.l2.weight, std=0.02)
        torch.nn.init.normal_(self.l3.weight, std=0.02)
 
    def forward(self, ids, mask, token_type_ids):
        
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.dropout(out)
 
        out = self.l0(out)
        out = self.l1(out)
        out = self.l2(out)
        logits = self.l3(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        #print(f"Start Logit size: {start_logits.size()}")
        #print(f"End Logit size: {end_logits.size()}")
          
        return start_logits, end_logits
```

## Loss

**1) Cross Entropy Loss**

```
def loss_fn(start_logits, end_logits, start_positions, end_positions):
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss
```

**2) Cross Entropy Loss with Label Smoothing**

Loss function to penalize far predictions more than close ones.
 
```
# Example Usage:- smooth_one_hot(torch.tensor([2, 3]), classes=10, smoothing=0.1)
def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
  """
  if smoothing == 0, it's one-hot method
  if 0 < smoothing < 1, it's smooth method
  """
  assert 0 <= smoothing < 1
  confidence = 1.0 - smoothing
  #print(f"Confidence:{confidence}")
  label_shape = torch.Size((true_labels.size(0), classes))
  #print(f"Label Shape:{label_shape}")
  with torch.no_grad():
    true_dist = torch.empty(size=label_shape, device=true_labels.device)
    #print(f"True Distribution:{true_dist}")
    true_dist.fill_(smoothing / (classes - 1))
    #print(f"First modification to True Distribution:{true_dist}")
    true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    #print(f"Modified Distribution:{true_dist}")
  return true_dist

def cross_entropy(input, target, size_average=True):
  """ Cross entropy that accepts soft targets
  Args:
        pred: predictions for neural network
        targets: targets, can be soft
        size_average: if false, sum is returned instead of mean
  """
  logsoftmax = nn.LogSoftmax(dim=1)
  if size_average:
      return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
  else:
      return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))
      
def loss_fn(start_logits, end_logits, start_positions, end_positions):
  smooth_start_positions = smooth_one_hot(start_positions, classes=config.MAX_LEN, smoothing=0.1)
  smooth_end_positions = smooth_one_hot(end_positions, classes=config.MAX_LEN, smoothing=0.1)

  start_loss = cross_entropy(start_logits, smooth_start_positions)
  end_loss = cross_entropy(end_logits, smooth_end_positions)
  total_loss = (start_loss + end_loss)
  
  return total_loss
```

## Training Schedule

get_cosine_schedule_with_warmup

get_linear_schedule_with_warmup

https://huggingface.co/transformers/main_classes/optimizer_schedules.html

## Optimizer

AdamW
