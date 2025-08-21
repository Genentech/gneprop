### Checkpoints on s1b dataset

Initial dataset used for model validation and optimization.
Notice that, given randomness, differences in library versions and GNEprop library itself,
results are slightly different compared to those reported in the manuscript, but the overall trends
are consistent. 20 folds each.

| Checkpoint      | Comment                                   | AUPRC | AUROC |   F1 |
| :-------------- | :---------------------------------------- | ----: | ----: | ---: |
| `20250801-153038` | GNEprop (no pretraining, no RDKit features)        | 0.486 | 0.807 | 0.456 |
| `20250801-154530` | GNEprop (no pretraining, + RDKit features)         | 0.497 | 0.825 | 0.433 |
| `20250801-150551` | GNEprop (+ pretraining, no RDKit features)         | 0.558 | 0.857 | 0.487 |
| `20250801-182534` | GNEprop (+ pretraining, + RDKit features)          | 0.560 | 0.848 | 0.482 |

We also release checkpoint `20250817-181637` which includes meta-learning-based fine-tuning on top of the
best model (not reported in the manuscript), further improving it (AUPRC = 0.560, AUROC = 0.873, F1 = 0.493)

### Checkpoints on GNEtolC dataset

Notice that GNEprop has not been explicitly hyper-optimized or evaluated on this dataset. 8 folds each.

* `20250819-085119`: GNEprop trained on GNEtolC dataset with scaffold splitting
* `20250819-093608`: GNEprop trained on GNEtolC dataset with scaffold-cluster splitting

### Self-supervised checkpoint
* `20210827-082422`: self-supervised checkpoint

### GNEprop trained on HTS dataset
* `20250811-202022`: [TO BE ADDED] GNEprop trained on the full HTS dataset (95/5 random splitting). 8 folds.