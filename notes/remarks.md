imbs lstm example


Loading data...
25000 train sequences
25000 test sequences
Pad sequences (samples x time)
x_train shape: (25000, 80)
x_test shape: (25000, 80)
Train on 25000 samples, validate on 25000 samples


batch size on P2 instance
size 32 - 113sec/epoch
size 256 - 16 sec/epoch
size 1024- 7 sec/epoch
size 4096- 5 sec/epoch
size 6144- 5 sec/epoch
size 8192- Resource exhausted: OOM when allocating tensor with shape[8192,80,512]
size 25000- Resource exhausted: OOM when allocating tensor with shape[2000000,128]

batch size on laptop CPU
size 32 - 227 - 253 sec/epoch
size 256 - 124 - 128 sec/epoch
size 1024 - 94 - 98 sec/epoch
size 4096 - 81 - 82 sec/epoch
size 6144 - 75  - 81 sec/epoch
size 8192 - fish: “python src/imdb_lstm.py” terminated by signal SIGKILL (Forced quit)



conclusion: load as many batches as you can
imbd 4096 is good starting point
