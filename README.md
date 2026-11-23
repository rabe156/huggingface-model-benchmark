# Hugging Face Sentiment Model Benchmark

A reproducible, CPU-oriented comparison of two pretrained Transformer models on the **Stanford Sentiment Treebank 2 (SST-2)** validation set. The project evaluates predictive quality, analyzes classification mistakes, and measures inference throughput at different batch sizes using PyTorch and Hugging Face Transformers.

**Models compared**

- **DistilBERT:** [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
- **BERT:** [`textattack/bert-base-uncased-SST-2`](https://huggingface.co/textattack/bert-base-uncased-SST-2)

**Dataset:** [`stanfordnlp/sst2`](https://huggingface.co/datasets/stanfordnlp/sst2), validation split (872 labeled examples). SST-2 is a binary sentiment-classification task: `0 = NEGATIVE`, `1 = POSITIVE`.

> **Scope:** Inference and evaluation only. No model training or fine-tuning is performed.

## Key findings

The following numbers are from the completed 872-example experiment, not illustrative estimates.

### Classification quality

| Model | Accuracy | Precision | Recall | F1-score | Correct / 872 |
|---|---:|---:|---:|---:|---:|
| DistilBERT | 91.06% | 89.78% | 93.02% | 91.37% | 794 |
| BERT | **92.43%** | **91.81%** | **93.47%** | **92.63%** | **806** |

BERT achieved **1.38 percentage points** higher accuracy, correctly classifying **12 more examples**. This validation-set comparison does not, by itself, establish statistical significance or performance on a different domain.

### Inference performance

Measured over the same 872 examples at each batch size. Execution times are the median of **three timed repetitions** per configuration, after a warm-up pass.

| Model | Batch size | Median time (s) | Throughput (samples/s) | Amortized time (ms/sample) |
|---|---:|---:|---:|---:|
| DistilBERT | 1 | 63.051 | 13.83 | 72.31 |
| DistilBERT | 4 | 49.674 | 17.55 | 56.97 |
| DistilBERT | **8** | **45.763** | **19.05** | **52.48** |
| DistilBERT | 16 | 46.369 | 18.81 | 53.18 |
| BERT | 1 | 116.811 | 7.47 | 133.96 |
| BERT | **4** | **83.778** | **10.41** | **96.08** |
| BERT | 8 | 86.310 | 10.10 | 98.98 |
| BERT | 16 | 92.926 | 9.38 | 106.57 |

At their best measured batch sizes, DistilBERT delivered **approximately 1.83× the throughput** of BERT. The optimal observed batch size was **8 for DistilBERT** and **4 for BERT** on the tested setup. These timings are environment-dependent; detailed CPU and memory specifications were not recorded in the supplied results.

**Interpretation:** BERT offers the higher observed classification quality; DistilBERT offers higher processing capacity. Selection should depend on acceptable error rates, workload, latency requirements, and deployment resources. `ms/sample` here is total execution time divided by dataset size, **not** per-request latency under production concurrency.

## Features

- Load a fixed subset of SST-2 validation examples.
- Load Hugging Face tokenizers and pretrained sequence-classification models.
- Run batched inference with dynamic padding and truncation.
- Compute accuracy, precision, recall, F1-score, and a confusion matrix.
- Identify false positives, false negatives, and high-confidence errors.
- Benchmark end-to-end inference over configurable batch sizes and repetitions.
- Export side-by-side CSV comparisons and error reports.
- Create accuracy/F1 and throughput comparison charts with Matplotlib.

## Repository layout

The module names below match the imports in the project's integration notebook.

```text
huggingface-model-benchmark/
├── src/
│   ├── __init__.py
│   ├── dataset.py           # load_data
│   ├── model.py             # load_model
│   ├── inference.py         # predict
│   ├── evaluation.py        # compute_metrics, analyze_errors
│   ├── benchmark.py         # benchmark_model
│   └── reporting.py         # comparison, exports, charts
├── results/
│   ├── model_comparison.csv
│   ├── performance_comparison.csv
│   ├── distilbert_errors.csv
│   ├── bert_errors.csv
│   └── figures/
│       ├── model_metrics.png
│       └── throughput_comparison.png
├── main.ipynb               # experiment notebook (rename if needed)
├── requirements.txt         # optional dependency lock/list
├── .gitignore
└── README.md
```

The `src/` modules must be alongside the notebook for the imports to work. The result files are produced when the experiment is executed; include them in the repository if you want visitors to inspect the measured outputs. The supplied notebook was named `main(20261123-171013).ipynb`; renaming it to `main.ipynb` is recommended for publication. A standalone `main.py` can be added later if desired.

## Setup

**Prerequisites:** A compatible Python environment, internet access on the first run to fetch the dataset/model checkpoints, and sufficient RAM for the models. No GPU is required by the current CPU-oriented implementation.

```bash
# From the project root
python -m venv .venv
```

Activate the environment:

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (Command Prompt):** `.venv\Scripts\activate.bat`
- **Linux/macOS:** `source .venv/bin/activate`

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch transformers datasets scikit-learn pandas numpy matplotlib jupyter
```

If you create a version-pinned `requirements.txt`, prefer using:

```bash
python -m pip install -r requirements.txt
```

## Run the experiment

Open the notebook from the repository root:

```bash
jupyter notebook
```

Then open `main.ipynb` and run all cells. The integration notebook imports modules from `src/`, configures the experiment, evaluates each model, combines the results, and writes CSV/PNG reports.

For a **quick smoke test**, use:

```python
MAX_EXAMPLES = 20
INFERENCE_BATCH_SIZE = 8
BENCHMARK_BATCH_SIZES = [8]
BENCHMARK_REPETITIONS = 1
```

For the **full benchmark**, use:

```python
MAX_EXAMPLES = 872
INFERENCE_BATCH_SIZE = 8
BENCHMARK_BATCH_SIZES = [1, 4, 8, 16]
BENCHMARK_REPETITIONS = 3
```

The small configuration was reported to pass an end-to-end smoke test, including checks for the two models' outputs and the six expected report files. Formal `pytest` unit tests are **deferred**, not claimed as completed.

## Evaluation protocol

1. Load the SST-2 **validation** split and use the same examples for both models.
2. Tokenize the text with each checkpoint's own tokenizer; perform batched inference without gradient calculations.
3. Confirm class-index semantics before scoring: SST-2 label `0` means negative and `1` means positive. Some checkpoints expose generic `LABEL_0` / `LABEL_1` names; do **not** infer class meaning from the name alone.
4. Compare predicted class indices against ground truth using accuracy, precision, recall, F1, and a confusion matrix.
5. Collect misclassified examples and record the predicted class and confidence for qualitative analysis.
6. For each batch size, warm up the model, then time three full inference passes using `time.perf_counter()` and report their median.
7. Compute `throughput = number_of_examples / median_time_s` and `ms_per_sample = 1000 / throughput`.

The benchmark measures the project's inference path, including tokenization and postprocessing. Model download/loading is outside the timed loop. Results will change across CPUs, software versions, sequence-length distributions, and resource contention. Validation-set scores should not be interpreted as performance guarantees on unseen production text.

## Generated reports

| File | Contents |
|---|---|
| `results/model_comparison.csv` | Accuracy, precision, recall, and F1 by model |
| `results/performance_comparison.csv` | Median runtime, throughput, and ms/sample by batch size |
| `results/distilbert_errors.csv` | DistilBERT's misclassified validation examples |
| `results/bert_errors.csv` | BERT's misclassified validation examples |
| `results/figures/model_metrics.png` | Grouped accuracy/F1 comparison |
| `results/figures/throughput_comparison.png` | Throughput vs. batch size |

The `.gitignore` deliberately **does not exclude `results/`**, so the benchmark tables and figures can be committed as portfolio evidence. It does exclude local caches, downloaded model weights, virtual environments, editor files, and secrets.

## Limitations and future work

- **Validation, not an independent held-out test:** Checkpoint development and prior exposure should be considered before drawing generalization conclusions.
- **Hardware details not captured:** Reproduction of exact runtimes requires recording CPU model, RAM, OS, thread settings, and dependency versions.
- **No confidence calibration analysis:** Confidence scores are useful for inspecting errors, but they are not necessarily calibrated probabilities.
- **No production load test:** Throughput on a static dataset is distinct from request latency, concurrent serving, and tail latency.
- **Automated unit tests pending:** The integration smoke test passed; `pytest` coverage was intentionally postponed.
- **Potential extensions:** Add unit tests, preserve environment metadata, compare memory use and peak RAM, analyze confusion matrices and confidence calibration, and add a command-line entry point.

## Learning objectives

This project was built as the **Week 6 (Hugging Face & pretrained models)** portfolio project in an AI/ML retraining curriculum. It emphasizes practical use of model checkpoints, dataset handling, evaluation, performance benchmarking, and reproducible engineering workflows before the subsequent fine-tuning module.
