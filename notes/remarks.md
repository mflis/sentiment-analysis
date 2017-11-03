# Batch size

Experiments on imdb lstm example

### preparations
```
Loading data...
25000 train sequences
25000 test sequences
Pad sequences (samples x time)
x_train shape: (25000, 80)
x_test shape: (25000, 80)
Train on 25000 samples, validate on 25000 samples
```


### P2 instance (GPU)

|batch size|time \[sec/epoch\] |
|----------|-------------|
|32|  113|
|256| 16 |
|1024| 7 |
|4096|5 |
|6144|5 |
|8192|Resource exhausted: OOM when allocating tensor with shape\[8192,80,512\]
|25000|Resource exhausted: OOM when allocating tensor with shape\[2000000,128\]

### laptop CPU

|batch size|time \[sec/epoch\] |
|----------|-------------|
|32|227 - 253 |
|256|124 - 128 |
|1024|94 - 98 |
|4096|81 - 82 |
|6144|75  - 81 |
|8192|fish: “python src/imdb_lstm.py” terminated by signal SIGKILL (Forced quit)



## Conclusion
-  load as many batches as you can
- batch size 4096 is good starting point
- open question: can too big batch size affect network's ability to learn?



# Activation functions
experiments on tf-idf + simple network example

https://medium.com/towards-data-science/activation-functions-and-its-types-which-is-better-a9a5310cc8f
- `relu` is good but only as intermediate layer (when used as output/last activation it performs worse than random guess)(it's not designed to work with binary crossentropy)
- `softmax` (as the last activation) - for some reason it doesn't learn anything. It always predicts 1, AUC is 0.5
- open question: why soft max performed so badly?


# weight initializations
- change initializer from RandomNormal to `glorot_uniform (Xavier)` made huge difference (from AUC aroud 0.5 to AUC about 0.8)


# feed-forward network

https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
- rule of thumb for hidden layer size:
  - one layer is usually enough (2 or more do not give much gains)
  - size of hidden layer (input layer + output layer)/2

https://datascience.stackexchange.com/questions/806/advantages-of-auc-vs-standard-accuracy