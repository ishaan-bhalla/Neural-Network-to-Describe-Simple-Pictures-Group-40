<div align="center">

# Neural Networks to Describe Simple Pictures

**How much does the *shape* of a caption decide what a model can learn? Three image-to-text architectures, three synthetic datasets, and a 32B vision-language model as a reference.**

[![Live demo](https://img.shields.io/badge/Live_demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ishaan-bhalla-neural-network-to-describe-simple-pict-app-u8tfxe.streamlit.app/)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![MSc group project](https://img.shields.io/badge/MSc_group_project-University_of_Bristol-5f6368?style=for-the-badge)

<img src="docs/figures/predictions.png" alt="Five test images with the MultiHead CNN's predicted captions" width="95%">

<sub>The first test image of each dataset, with the MultiHead CNN's prediction (not hand-picked). The middle board is the model's one miss out of 30.</sub>

</div>

## Key results

- **Tic-tac-toe: 98.9% exact captions on 1,000 unseen boards** (95% CI 98.0-99.4%) and 99.89% of individual cells correct, with a 2.5M-parameter CNN that predicts each cell separately. On the original 30-board test split it scored 96.67% (29/30).
- **It matches or beats a 32B vision-language model, about 47,000 times faster.** On the original 30 boards both models scored 29/30. On 100 new boards the CNN got 100/100 and Qwen3-VL 32B (zero-shot, local) 94/100. The CNN takes 1.3 ms per board on a laptop CPU; Qwen3-VL takes about 59 s per board on the laptop GPU.
- **It knows when it is unsure.** If the CNN abstains when any cell's confidence is below 0.8, it answers 85.8% of the 1,000 test boards and gets every one of them right.
- **Caption structure decides difficulty.** The same multi-head design reaches only 25% on Shapes and 4% on Numbers, where one wrong attribute out of seven makes the whole caption wrong. On Shapes, a retrieval model (37%) beats it.

## Contents

1. [The question](#1-the-question)
2. [Datasets](#2-datasets)
3. [Models](#3-models)
4. [Results](#4-results)
5. [What we learned and limitations](#5-what-we-learned-and-limitations)
6. [Live demo](#6-live-demo)
7. [Run it](#7-run-it)
8. [Team and my contribution](#8-team-and-my-contribution)

## 1. The question

Image captioning is usually treated as generating a sentence word by word. But many captions follow a fixed template: "X is in the top left", "a small red triangle is left of a blue circle". We compared three ways of producing a caption, each treating it differently:

| Caption treated as | Model | How it produces a caption |
|---|---|---|
| A sentence to retrieve | **Embedding CNN** | Matches the image to the closest caption in a pool |
| A sequence of words | **CNN + LSTM** | Generates the caption one word at a time |
| A set of slots to fill | **MultiHead CNN** | Predicts each attribute separately, then fills a template |

## 2. Datasets

All images are generated with PIL from a fixed random seed, so every result is reproducible. Splits are 80% train, 10% validation and 10% test.

| Dataset | Images | Size | Caption template | Attributes |
|---|---|---|---|---|
| **Tic-tac-toe** | 300 | 256×256 | `X is in [cells]; O is in [cells]` | 9 cells (X, O or empty) |
| **Shapes** | 1,000 | 256×256 | `a [size] [colour] [shape] is [relation] a [size] [colour] [shape]` | 7 |
| **Numbers** | 1,000 | 128×128 | `a [size] [colour] [number]` | 7 (size, colour, length, 4 digits) |

## 3. Models

**Embedding CNN (retrieval).** Image → CNN → 384-dimensional vector, trained to sit close to the caption's Sentence-BERT embedding (`all-MiniLM-L6-v2`). At test time it returns the nearest caption by cosine similarity.

**CNN + LSTM (generation).** Image → CNN encoder → 256-dimensional feature → LSTM decoder. Trained with teacher forcing, decoded greedily.

**MultiHead CNN (structured prediction).** Image → 4-block CNN → 512-dimensional feature → one classification head per attribute (9 for tic-tac-toe, 7 for Shapes and Numbers). The loss is the sum of the heads' cross-entropies, and the caption is rebuilt from the template. It keeps a 4×4 spatial grid (`AdaptiveAvgPool2d(4)`) instead of pooling to a single vector, so the heads can still tell where things are.

[![Architecture diagram of ishaan-bhalla/neural-network-to-describe-simple-pictures-group-40](https://gitdiagram.com/ishaan-bhalla/neural-network-to-describe-simple-pictures-group-40/diagram.png)](https://gitdiagram.com/ishaan-bhalla/neural-network-to-describe-simple-pictures-group-40?utm_source=readme&utm_medium=picture)

*Architecture diagram (GitDiagram; click to explore it interactively).*

## 4. Results

**Exact-caption (sentence) accuracy on the test set**

| Dataset | Embedding CNN | CNN + LSTM | MultiHead CNN |
|---|---|---|---|
| **Tic-tac-toe** (30 test images) | 20% | not recorded (74.5% of words correct) | **96.67%** (29/30) |
| **Shapes** (100) | **37%** | 0% (60.9% of words correct) | 25% (best run, 50 epochs); 7% for the 10-epoch weights in the demo |
| **Numbers** (100) | 5% | not recorded (56.0% of words correct) | 4% |

**Vision-language model reference (tic-tac-toe).** GPT-3.5 can't read images, so we used an open vision model instead. Qwen3-VL 32B ran locally through Ollama, zero-shot, at temperature 0, and was asked for the nine cells as JSON. It ran on the same 30 seeded test boards.

| Model | Exact captions | Cells correct |
|---|---|---|
| MultiHead CNN (ours, 2.5M parameters) | **29/30 (96.67%)** | **99.26%** (268/270) |
| Qwen3-VL 32B, zero-shot | 29/30 (96.67%) | 98.52% (266/270) |

With one miss each on 30 boards, this is a tie, not a win. The larger evaluation below separates them.

### Larger held-out evaluation (1,000 boards)

Run on 2026-10-23 with the committed weights the demo serves (no retraining). Scripts and raw results are in [`Evaluation/`](Evaluation/).

**Test set.** `Evaluation/make_tictactoe_heldout.py` uses the project's generator and renderer with a new seed (2026) to draw 1,500 distinct boards. None of them has the same cell layout as any of the 300 original boards (train, val or test), and the script checks this. They are shuffled into a 500-board validation split, used only to choose the abstention threshold, and a 1,000-board test split. Because 2-mark layouts are few and most are already in the original data, the new set has more crowded boards than the original (test: 20 boards with 2 marks, 121 with 3, 261 with 4, 302 with 5, 296 with 6). Of the original 30 test boards, 6 have a layout that also appears in training.

**Accuracy (MultiHead CNN).** Sentence accuracy CIs are Wilson 95%; cell accuracy CIs are a 95% bootstrap over boards (cells on one board are not independent).

| Test set | Boards | Exact captions | 95% CI | Cells correct | 95% CI |
|---|---|---|---|---|---|
| Original test split | 30 | 29 (96.67%) | 83.3-99.4% | 268/270 (99.26%) | 97.8-100% |
| New held-out test | 1,000 | **989 (98.9%)** | 98.0-99.4% | **8,989/9,000 (99.89%)** | 99.80-99.94% |

**Abstention ("I don't know").** Confidence is the lowest top-class softmax probability across the 9 cell heads. The CNN answers only when that is at or above a threshold.

| Threshold | Answered (test) | Accuracy on answered | Wrong boards caught |
|---|---|---|---|
| none | 1,000 (100%) | 98.9% | 0 of 11 |
| 0.6 | 980 (98.0%) | 99.49% (975/980) | 6 of 11 |
| 0.7 | 933 (93.3%) | 99.68% (930/933) | 8 of 11 |
| 0.8 | 858 (85.8%) | 100% (858/858) | 11 of 11 |
| 0.9 | 644 (64.4%) | 100% (644/644) | 11 of 11 |
| 0.95 | 369 (36.9%) | 100% (369/369) | 11 of 11 |

Thresholds were picked on the validation split, not the test split. The rule set beforehand was "the lowest threshold with no errors on validation", which gives 0.95 and keeps only 36.9% of test boards. A second rule, "the lowest threshold with at least 99.5% accuracy on validation", gives 0.6 (98.0% answered, 99.49% correct); it was added after seeing the first. The 0.8 row is the best trade-off on the test set, but validation still had one error there (434/435), so it is not a validation-chosen setting. The model is under-confident: almost no board has every cell above 0.99.

**Speed and size against Qwen3-VL 32B.** Same prompt and settings as the 30-board baseline (JSON output, temperature 0, thinking off). Qwen3-VL was run on the first 100 boards of the shuffled test split (a random sample); the run was capped at 100 boards because each takes about a minute.

| | MultiHead CNN | Qwen3-VL 32B (Ollama, Q4_K_M) |
|---|---|---|
| Exact captions, same 100 boards | **100/100** (Wilson 95% CI 96.3-100%) | 94/100 (87.5-97.2%) |
| Cells correct, same 100 boards | **100%** | 97.67% |
| Latency per board, p50 / p95 | **1.26 ms / 1.41 ms** (CPU, batch 1, n = 1,000) | 59.2 s / 113.1 s (Apple GPU, n = 100) |
| Parameters | 2.5M (2,500,891) | 33.4B (13,355x) |
| Model file | 10.0 MB | 20.9 GB (2,086x) |

On those 100 boards Qwen3-VL missed 6 that the CNN got right, and the CNN missed none that Qwen3-VL got right (exact McNemar test, p = 0.031). The CNN time is end to end (read the PNG, resize, forward pass, softmax) after 50 warm-up images; the forward pass alone is 0.82 ms p50. The Qwen3-VL time is one API call with the model already loaded, after one untimed warm-up call. Qwen3-VL's time varies with how many tokens it emits (mean 327 for a short JSON answer); its slowest board took 234 s. Machine: Apple M3 Pro, 36 GB RAM, macOS, PyTorch 2.12.1 (6 CPU threads). The CNN was timed with Ollama idle; the Qwen3-VL run shared the machine with some idle Docker containers (1-minute load average 0.3-5.0), so its times may be slightly high.

**Fallback cascade.** Boards below a confidence threshold go to Qwen3-VL; the rest keep the CNN's answer. On the 100-board sample:

| Threshold | Sent to Qwen3-VL | Exact captions | Mean latency per board |
|---|---|---|---|
| no cascade (CNN only) | 0% | 100/100 | 1.3 ms |
| 0.6 | 2% | 100/100 | 1.4 s |
| 0.7 | 4% | 100/100 | 2.9 s |
| 0.8 | 11% | 99/100 | 9.2 s |
| 0.95 | 60% | 96/100 | 41.9 s |

Because the CNN is more accurate than Qwen3-VL here, sending boards to the bigger model adds latency without adding accuracy; on this sample it only loses captions. The full 1,000-board cascade could not be scored, because Qwen3-VL answered only 2 of the 20 boards the 0.6 threshold would route (on all 1,000, a 0.6 cascade would land between 97.5% and 99.5%). For this task, abstaining is the better use of the confidence score than escalating.

**Per-attribute accuracy, MultiHead CNN**

| Tic-tac-toe | | Shapes (best run) | | Numbers | |
|---|---|---|---|---|---|
| 7 of 9 cells | 100% | Object 1 size | 79% | Length | 97% |
| top right, bottom right | 96.7% | Object 1 colour | 65% | Digit 4 | 88% |
| | | Object 1 shape | 64% | Colour | 82% |
| | | **Relation** | **50%** | Digit 3 | 64% |
| | | Object 2 size | 83% | Digit 1 | 60% |
| | | Object 2 colour | 59% | Digit 2 | 51% |
| | | Object 2 shape | 60% | **Size** | **47%** |

## 5. What we learned and limitations

- **When the caption is really a set of slots, predict the slots.** Tic-tac-toe has nine independent cells and no grammar, and the multi-head model is near-perfect there.
- **Exact-match accuracy punishes long templates.** On Shapes, each attribute is right 50-83% of the time, but all seven are right together only 25% of the time. Per-attribute accuracy is the fairer measure there.
- **Relations are the bottleneck.** "Above / below / left of / right of" is 50% on Shapes, against chance of 25%. The model has to compare two objects' positions, which independent heads don't do well.
- **Small test sets.** The original tic-tac-toe split has 30 test images, so one error moves accuracy by 3.3 points, and 6 of the 30 share a layout with a training board. The 1,000-board held-out set above gives a tighter estimate (98.9%, 95% CI 98.0-99.4%).
- **Synthetic data only.** The models have never seen real photographs and won't generalise to them.
- **The CNN + LSTM wasn't pretrained.** It learns word order well but rarely gets a whole caption right on the larger datasets.

## 6. Live demo

**[Open the demo](https://ishaan-bhalla-neural-network-to-describe-simple-pict-app-u8tfxe.streamlit.app/)**. Pick a dataset, draw a random test image or upload your own, and compare the MultiHead CNN's and CNN + LSTM's captions, with each head's confidence. The app sleeps when nobody has used it for a while; click "Yes, get this app back up" and it starts in about 30 seconds.

## 7. Run it

Tested with Python 3.13.

```bash
git clone https://github.com/ishaan-bhalla/Neural-Network-to-Describe-Simple-Pictures-Group-40.git
cd Neural-Network-to-Describe-Simple-Pictures-Group-40
pip install -r requirements.txt          # demo only
pip install -r requirements-train.txt    # adds what training needs

# Generate the datasets (seeded)
python src/data/generate_shapes.py
python src/data/generate_numbers.py --num_samples 1000
python src/data/generate_tictactoe.py --num-samples 300

# Train (weights go to artifacts/{dataset}/{model}/best_model.pth)
python -m src.training.train_multihead_cnn --dataset tictactoe --epochs 50
python -m src.training.train_cnn_lstm --dataset shapes
python -m src.training.train_embedding_model --dataset-name shapes

# Larger tic-tac-toe evaluation (no retraining; the Qwen3-VL step needs Ollama with qwen3-vl:32b)
python Evaluation/make_tictactoe_heldout.py
python Evaluation/eval_multihead_tictactoe.py
python Evaluation/qwen3vl_tictactoe.py --split test --ids first:100 --out Evaluation/qwen3vl_tictactoe_runs_$(date +%F).jsonl
python Evaluation/cascade_tictactoe.py

# Run the demo
streamlit run app.py
```

## 8. Team and my contribution

An MSc group project at the University of Bristol by Group 40: Ishaan Bhalla, [FreyaYZeng](https://github.com/FreyaYZeng), [rmkhan1999](https://github.com/rmkhan1999), [simrithcprince16](https://github.com/simrithcprince16) and [zxxz1203](https://github.com/zxxz1203). The team repository is [rmkhan1999/Neural-Network-to-Describe-Simple-Pictures-Group-40](https://github.com/rmkhan1999/Neural-Network-to-Describe-Simple-Pictures-Group-40); this fork adds the deployed demo and this write-up.

**My contribution (Ishaan Bhalla)**

- **Led model development:** wrote the model and training code for all three architectures and the simple CNN baseline (`src/models/`, `src/training/`).
- **Data loading:** `get_metadata_files`, which loads either one metadata file (Shapes, Numbers) or three split files (tic-tac-toe) through the same interface.
- **Inference and demo:** `src/inference/` and the Streamlit app, which I deployed to Streamlit Community Cloud.
- **Vision-model baseline:** after the project, ran Qwen3-VL 32B on the tic-tac-toe test set to replace a cited GPT-3.5 figure with a measured comparison.
- **Larger evaluation:** a 1,000-board held-out test set with confidence intervals, confidence-based abstention, latency against Qwen3-VL and a fallback cascade (`Evaluation/`).
