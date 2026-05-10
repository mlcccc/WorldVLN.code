# InfinityStar Backbone Training

This directory is a trimmed training release centered on a single InfinityStar backbone finetuning entrypoint.

## What Is Included

The current `backbone/` package keeps the code required to launch training from original sharded base weights:

- `scripts/train_from_base.sh`: the only supported training launcher
- `train.py`: distributed training entrypoint
- `infinity/`: model, trainer, dataset, schedule, and utility code used by training
- `TRAINING.md`: detailed setup and launch guide
- `requirements.txt`: Python dependencies for this trimmed release

This directory does not currently ship inference entrypoints, demo assets, or web demo code.

## Installation

1. Use a Python environment compatible with `torch>=2.5.1`.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

The supported launcher is:

```bash
bash scripts/train_from_base.sh
```

For required checkpoint layout, JSONL schema, environment variables, and launch examples, see `TRAINING.md`.

## Repository Layout

```text
backbone/
|-- README.md
|-- TRAINING.md
|-- requirements.txt
|-- train.py
|-- scripts/
|   `-- train_from_base.sh
`-- infinity/
```

## Scope

This `backbone/` subtree is intended to be a focused training package. Files or workflows from earlier full-repository snapshots that are not part of the current single-script training path are intentionally not documented here.

## Citation

If this release is useful in your research, please cite:

```bibtex
@misc{InfinityStar,
  title={InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation},
  author={Jinlai Liu and Jian Han and Bin Yan and Hui Wu and Fengda Zhu and Xing Wang and Yi Jiang and Bingyue Peng and Zehuan Yuan},
  year={2025},
  eprint={2511.04675},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2511.04675}
}
```

## License

This project is licensed under the MIT License. See `LICENSE`.
