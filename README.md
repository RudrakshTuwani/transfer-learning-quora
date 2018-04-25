# Predicting duplicate question pairs on Quora using transfer learning.

Performance of recently proposed tranfer learning techniques for Natural Language Processing (NLP) applications 
([FitLaM](https://arxiv.org/abs/1801.06146) and [USE](https://arxiv.org/abs/1803.11175))
are compared on the [Quora Duplicate Question Pair Prediction Challenge](https://www.kaggle.com/c/quora-question-pairs).
The effect on performance by varying dataset size is also investigated. 

# Validation Set

## 1. Negative Log Loss
| Model/Training Set Size | 100  | 1000  | 10000  | 50000  | 100000 | All |
|---|---|---|---|---|---|---|
| ArCos USE  |   |   |   | | |0.537 |
