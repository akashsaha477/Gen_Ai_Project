# runs

One folder per training run, grouped by experiment, variant and kind of run:

```
runs/
  m1_input_representation/
    A_short_window/              <- one folder per Module 1 run of variant A
    B_long_window_bandpass/ ...
  ptbxl_pretraining/
    D_short_window_125hz/
      pretrain/                  <- pretraining on PTB-XL, one run per seed
      scratch/                   <- MIT-BIH from scratch, 6 runs
      finetune/                  <- MIT-BIH from the pretrained weights, 6 runs
```

Inside each run folder:

| file | content |
|---|---|
| `config.yaml` | every setting of the run, including seed, learning rate and the commit |
| `env.json` | Python and package versions, hardware, wall-clock, checkpoint hash |
| `train_log.csv` | one line per epoch: training loss and validation scores |
| `metrics_val.json` | validation metrics of the best epoch, with the confusion matrix |
| `preds_val.npz`, `checkpoints/best.pt` | predictions and weights; not committed (too large) |

`results/runs_index.csv` lists every run with its folder. Do not edit or delete run folders; a new
run always gets a new folder.
