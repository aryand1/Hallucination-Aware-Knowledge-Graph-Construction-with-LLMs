**PERSEUS** stands for **PERceptual Semantic Extraction & Unified System**. It is a research pipeline for constructing knowledge graphs and ontologies from text, image, and audio inputs while reducing hallucinated or unsupported triples produced by large language models.

The project focuses on a practical problem in LLM-assisted knowledge graph construction: LLMs can extract fluent subject-predicate-object triples that appear plausible but are not entailed by the source material. PERSEUS addresses this by combining constrained triple extraction, evidence grounding, repair, Natural Language Inference (NLI) validation, entity clustering, and ontology construction.

> **Repository status:** this repository contains public-safe scaffolds, evaluation notebooks, benchmark templates, multimodal preprocessing utilities, sample abstract data, and result artifacts. Some core model prompts, proprietary logic, datasets, paths, and implementation details are intentionally redacted or represented as templates/stubs.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Research Motivation](#research-motivation)
- [Pipeline Architecture](#pipeline-architecture)
- [Repository Contents](#repository-contents)
- [Core Modules](#core-modules)
- [Notebooks and Experiments](#notebooks-and-experiments)
- [Data and Result Files](#data-and-result-files)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Expected Workflow](#expected-workflow)
- [Evaluation Summary](#evaluation-summary)
- [Configuration Notes](#configuration-notes)
- [Limitations](#limitations)
- [Recommended Repository Structure](#recommended-repository-structure)
- [Citation](#citation)
- [License and Use](#license-and-use)

---

## Project Overview

PERSEUS is designed to transform raw multimodal inputs into a more trustworthy knowledge graph by requiring extracted triples to pass source-grounded validation before being stored or used for ontology construction.

At a high level, PERSEUS supports:

- **Text inputs**: research abstracts, technical documents, benchmark sentences, or raw text corpora.
- **Image inputs with text**: screenshots, scanned documents, slides, forms, and document images processed through OCR.
- **Image inputs without text**: natural images processed through image captioning models.
- **Audio inputs**: speech data transcribed into text before extraction.
- **Triple repair**: realignment of extracted subject/object spans to the original source sentence.
- **Evidence-grounded validation**: lexical checks, BM25 retrieval, context-window escalation, and NLI verification.
- **Ontology construction**: validated triples are used to induce classes, property assertions, and candidate axioms.
- **Auditability**: accepted and rejected triples can be logged with confidence scores, error tags, and supporting evidence.

---

## Research Motivation

Knowledge graphs and ontologies are useful for structured reasoning, retrieval, validation, and semantic search. However, manual KG construction is slow and expensive. LLMs can accelerate triple extraction, but raw LLM extraction is risky because the model may introduce unsupported facts.

PERSEUS is built around the following design principle:

> A triple should not enter the knowledge graph simply because an LLM generated it. It should enter only when the source text provides enough evidence to support it.

The attached manuscript frames this as a hallucination-aware KG construction problem and evaluates PERSEUS against raw LLM extraction and traditional OpenIE baselines.

---

## Pipeline Architecture

The PERSEUS pipeline is organized into five major stages.

```text
Raw multimodal input
        |
        v
[1] Multimodal input acquisition
        |
        v
[2] Text preprocessing and sentence segmentation
        |
        v
[3] LLM-based triple extraction
        |
        v
[4] Triple repair and grounding validation
        |
        v
[5] Triple validation and ontology construction
        |
        v
Validated triples, knowledge graph, and ontology artifacts
```

### 1. Multimodal Input Acquisition

Inputs are normalized into textual form before triple extraction.

| Input type | Processing path | Example tools evaluated or referenced |
|---|---|---|
| Plain text | Direct text ingestion | Python text preprocessing |
| Image with text | OCR to text | EasyOCR, Tesseract, docTR |
| Image without text | Image captioning | BLIP, BLIP2, GIT, ViT-GPT-2 |
| Audio | Speech transcription | Whisper |

### 2. Text Preprocessing

Preprocessing cleans and segments text before extraction. The public utility module includes a minimal preprocessing baseline that:

- normalizes whitespace,
- removes simple boilerplate patterns,
- splits text into sentence-like segments,
- further segments clauses using lightweight rules,
- preserves provenance metadata such as source identifier and modality.

### 3. Triple Extraction

The research pipeline uses a constrained LLM extraction step to produce candidate triples in subject-predicate-object format. The public repository does **not** include private model prompts, model weights, or proprietary extraction logic. Instead, the included scaffold shows the expected interfaces and orchestration points.

### 4. Triple Repair

The manuscript describes a two-stage repair process:

- **R1 span canonicalization**: aligns extracted subjects and objects back to spans in the source sentence.
- **R2 grounding validation**: drops triples whose arguments or predicates cannot be grounded in source evidence.

Repair also preserves negation and modality cues so that speculative or negative claims are not converted into unsupported positive assertions.

### 5. Triple Validation and Ontology Construction

Validated triples pass through additional checks before being used for ontology construction:

- local lexical subject/object checks,
- local and context-window NLI,
- retrieval trigger policy,
- BM25 retrieval,
- retrieval-stage NLI,
- entity clustering,
- axiom induction,
- OWL/SHACL/OOPS-style verification checks,
- human-readable ontology finalization.

---

## Repository Contents

The uploaded repository materials include the following files.

| File | Purpose |
|---|---|
| `README.md` | Earlier README draft containing a broad paper-style overview. |
| `Pasted text(18).txt` | Manuscript LaTeX source for the paper, including motivation, methodology, evaluation, and results. |
| `Sample_Code_for_Hallucination.py` | Public dependency-free scaffold for a hallucination-aware KG pipeline. It exposes data models and orchestration hooks but omits private implementations. |
| `multimodal_conversion.py` | Public multimodal normalization and preprocessing utilities for text, image, and audio inputs. |
| `clustering_algorithms_.ipynb` | Notebook for entity embedding and clustering algorithm experimentation. The notebook appears partially malformed as JSON but its source text is still inspectable. |
| `Clustering_algo_results.docx` | Clustering output report comparing HDBSCAN, Affinity Propagation, Spectral Clustering, DBSCAN, and Density Peak Clustering. |
| `Image_descriptor_comparison (1).ipynb` | Image captioning benchmark notebook comparing GIT, BLIP, ViT-GPT-2, and BLIP2 variants. |
| `OCR_comp&results (1).ipynb` | OCR benchmark notebook comparing Tesseract, EasyOCR, Donut, Nougat, and docTR on document-image datasets. |
| `perseus_rerun_template.ipynb` | Redacted CaRB rerun template for extraction, repair, validation, evaluation, McNemar testing, bootstrap confidence intervals, and error analysis. |
| `perseus_ablation_template.ipynb` | Redacted ablation-study template for evaluating the contribution of repair, NLI, context windows, and retrieval. |
| `Abstract.zip` | Collection of RTF research abstracts used as sample source material. |

---

## Core Modules

### `multimodal_conversion.py`

This module provides public-safe utilities for converting heterogeneous inputs into text chunks and clause-level records.

#### Main data structures

| Class | Description |
|---|---|
| `RawInput` | Container for unprocessed input with `kind`, `payload`, and metadata. |
| `TextChunk` | Normalized textual unit produced from text, image OCR/captioning, or audio transcription. |
| `Clause` | Clause-level unit with provenance metadata for downstream extraction. |

#### Main classes

| Class | Responsibility |
|---|---|
| `MultimodalNormalizer` | Routes text, image, and audio inputs into textual chunks. OCR, captioning, and ASR are extension points. |
| `MinimalTriadPreprocessor` | Performs lightweight normalization and clause segmentation. |

#### Extension points

The following methods are intentionally abstract and must be implemented by the integrating application:

```python
def _run_ocr(self, image: Any) -> str:
    raise NotImplementedError

def _run_captioning(self, image: Any) -> str:
    raise NotImplementedError

def _run_asr(self, audio: Any) -> str:
    raise NotImplementedError
```

This design keeps the public module lightweight while allowing production systems to plug in OCR, captioning, or ASR backends.

### `Sample_Code_for_Hallucination.py`

This file is a public scaffold for the end-to-end hallucination-aware KG pipeline.

#### Data models

| Class | Description |
|---|---|
| `Triple` | Subject-predicate-object candidate fact. |
| `Evidence` | Candidate supporting sentence with retrieval-rank metadata. |
| `ValidationResult` | Verification decision for a triple, including acceptance status and entailment score. |

#### Pipeline constants

```python
RRF_K = 60
ENTAILMENT_THRESHOLD = 0.70
LEXICAL_ENTITY_CHECK = 0.95
```

#### Pipeline stages exposed by `EchoLLMPipeline`

| Method | Intended role |
|---|---|
| `preprocess()` | Normalize and segment source text. |
| `extract_triples()` | Call an instruction-following LLM to produce triples. Placeholder only. |
| `retrieve_evidence()` | Retrieve supporting sentences using lexical and dense retrieval. Placeholder only. |
| `verify()` | Apply lexical/NLI validation. Placeholder only. |
| `cluster_entities()` | Cluster entities to induce ontology classes. Placeholder only. |
| `build_ontology()` | Build a serializable ontology-like output from accepted triples. |
| `run()` | Orchestrate the full pipeline. |

The scaffold is intentionally runnable but does not perform real extraction or verification until private implementations are added.

---

## Notebooks and Experiments

### `perseus_rerun_template.ipynb`

A redacted template for rerunning PERSEUS on the CaRB benchmark. It is structured around the following workflow:

1. configure paths and runtime settings,
2. load Llama, NLI, SBERT, and spaCy models,
3. load and normalize CaRB ground truth,
4. extract raw LLM triples,
5. repair triples,
6. validate triples using lexical, NLI, window, and BM25 stages,
7. evaluate predictions with three-tier semantic matching,
8. compute McNemar tests and bootstrap confidence intervals,
9. perform error analysis and timing summaries.

The template contains redacted paths and placeholder functions, so it should be treated as a reproducibility skeleton rather than a plug-and-play notebook.

### `perseus_ablation_template.ipynb`

A redacted ablation-study template used to estimate the contribution of individual pipeline components. The notebook is organized around variants such as:

- baseline raw LLM output,
- repair enabled/disabled,
- local NLI enabled/disabled,
- context-window escalation enabled/disabled,
- retrieval enabled/disabled.

It is intended to produce ablation summaries and compare each variant against the raw LLM baseline.

### `Image_descriptor_comparison (1).ipynb`

This notebook benchmarks captioning models on a 200-sample image-caption dataset. Reported models include:

- `microsoft/git-base`,
- `microsoft/git-large`,
- `Salesforce/blip-image-captioning-base`,
- `Salesforce/blip-image-captioning-large`,
- `nlpconnect/vit-gpt2-image-captioning`,
- `Salesforce/blip2-opt-2.7b`.

Reported metrics include CLIPScore, semantic similarity, ROUGE-L, and METEOR.

Representative reported results:

| Model | CLIPScore | Semantic Similarity | ROUGE-L | METEOR |
|---|---:|---:|---:|---:|
| GIT-base | 0.2754 | 0.6604 | 0.4503 | 0.3103 |
| GIT-large | 0.2711 | 0.6576 | 0.4362 | 0.3005 |
| BLIP-base | 0.2870 | 0.7243 | 0.5473 | 0.4237 |
| BLIP-large | 0.2936 | 0.7675 | 0.5157 | 0.4992 |
| ViT-GPT-2 | 0.2931 | 0.7406 | 0.5660 | 0.4890 |
| BLIP2-OPT | 0.3008 | 0.7801 | 0.5903 | 0.4849 |

In these results, BLIP2-OPT performs strongest on CLIPScore, semantic similarity, and ROUGE-L, while BLIP-large is strongest on METEOR.

### `OCR_comp&results (1).ipynb`

This notebook compares OCR/document-understanding models across multiple experiments. Evaluated systems include:

- Tesseract,
- EasyOCR,
- Donut,
- Nougat,
- docTR.

Representative reported results include:

| System | WER | CER | Notes |
|---|---:|---:|---|
| Tesseract | 0.8873 | 0.6849 | Baseline OCR run. |
| EasyOCR | 0.7216 | 0.3227 | Stronger WER/CER than Tesseract in the reported run. |
| Donut | 1.0000 | 1.0000 | Weaker in the reported setup. |
| Nougat | 1.5698 | 1.5657 | Weaker in the reported setup. |
| docTR | 0.8604 | 0.2519 | Lower CER but higher WER than the previous EasyOCR champion in the reported comparison. |

The notebook also includes Hugging Face authentication and dataset availability checks. Some cells require a valid Hugging Face token or access to specific hosted datasets.

### `clustering_algorithms_.ipynb` and `Clustering_algo_results.docx`

The clustering materials evaluate entity clustering over flavonoid-related triples. Algorithms explored include:

- HDBSCAN,
- Affinity Propagation,
- Spectral Clustering,
- DBSCAN,
- Density Peak Clustering,
- hierarchical/agglomerative clustering in the broader manuscript context.

The result document reports clustering behavior over 31 unique entities. It shows that density-based methods can mark many entities as noise, while Affinity Propagation and Spectral Clustering provide broader entity coverage. The manuscript positions Spectral Clustering as the selected clustering method for ontology class induction.

---

## Data and Result Files

### `Abstract.zip`

The archive contains 39 RTF abstracts covering topics such as:

- artificial intelligence,
- natural language processing,
- healthcare AI,
- cybersecurity,
- quantum computing,
- autonomous vehicles,
- renewable energy,
- climate change,
- cultural heritage,
- robotics,
- biotechnology,
- e-commerce,
- virtual and augmented reality.

These abstracts are useful as sample documents for text extraction, triple generation, and human-curated evaluation.

### `Clustering_algo_results.docx`

This document contains printed clustering outputs, including:

- entity extraction summary,
- embedding generation summary,
- per-algorithm cluster assignments,
- noise/outlier lists,
- internal validation metrics such as Silhouette, Davies-Bouldin, and Calinski-Harabasz scores.

---

## Installation

The exact dependency set depends on which part of the repository you run. The public scaffolds require only standard Python libraries, while the notebooks require machine learning, NLP, OCR, and evaluation libraries.

### Minimum Python version

Python 3.10 or later is recommended.

### Minimal dependencies for public scaffolds

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

The two public Python files use mostly standard-library imports. No model inference dependencies are required to run the placeholder scaffold.

### Recommended research dependencies

Install the following when running the notebooks or filling in private implementations:

```bash
pip install \
  numpy \
  pandas \
  scikit-learn \
  scipy \
  torch \
  transformers \
  sentence-transformers \
  spacy \
  nltk \
  rdflib \
  owlready2 \
  pyshacl \
  rank-bm25 \
  statsmodels \
  matplotlib \
  tqdm \
  jiwer \
  pillow \
  evaluate \
  datasets
```

For OCR and document-image experiments:

```bash
pip install pytesseract easyocr python-doctr opencv-python-headless
```

System-level Tesseract may also be required:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

For spaCy preprocessing:

```bash
python -m spacy download en_core_web_sm
```

For notebooks that use Hugging Face-hosted models or gated datasets, authenticate first:

```bash
huggingface-cli login
```

---

## Quick Start

### 1. Run the public scaffold

```bash
python Sample_Code_for_Hallucination.py
```

Expected behavior: the script runs a smoke-test pipeline over placeholder text and prints an ontology summary. It will not extract real triples until private implementations are added.

### 2. Use the multimodal preprocessing utility

```python
from multimodal_conversion import RawInput, MultimodalNormalizer, MinimalTriadPreprocessor

class DemoNormalizer(MultimodalNormalizer):
    def _run_ocr(self, image):
        return "Example OCR text from image."

    def _run_captioning(self, image):
        return "Example image caption."

    def _run_asr(self, audio):
        return "Example transcript from audio."

raw_inputs = [
    RawInput(kind="text", payload="Urban agriculture provides food and benefits biodiversity.", metadata={"id": "sample_1"}),
    RawInput(kind="image", payload="path/to/image.png", metadata={"id": "sample_image"}),
]

normalizer = DemoNormalizer()
chunks = normalizer.normalize(raw_inputs)

preprocessor = MinimalTriadPreprocessor()
clauses = preprocessor.preprocess_chunks(chunks)

for clause in clauses:
    print(clause)
```

### 3. Run evaluation notebooks

Open the notebooks in Jupyter or Google Colab:

```bash
jupyter lab
```

Recommended order:

1. `Image_descriptor_comparison (1).ipynb`
2. `OCR_comp&results (1).ipynb`
3. `perseus_rerun_template.ipynb`
4. `perseus_ablation_template.ipynb`
5. `clustering_algorithms_.ipynb`

Before running benchmark notebooks, replace redacted paths and placeholders with your local paths, model names, and dataset locations.

---

## Expected Workflow

A typical PERSEUS experiment follows this flow:

1. **Prepare input corpus**
   - Use raw text, RTF abstracts, images, or audio.
   - Convert all modalities to text.

2. **Preprocess text**
   - Normalize characters and whitespace.
   - Remove boilerplate.
   - Segment into sentences or clauses.
   - Assign stable sentence/source identifiers.

3. **Extract candidate triples**
   - Run the configured LLM extractor.
   - Parse output into `[subject, predicate, object]` triples.
   - Remove malformed or duplicate triples.

4. **Repair triples**
   - Align subject and object spans to the source sentence.
   - Normalize predicates.
   - Preserve negation and modality cues.
   - Drop ungroundable triples.

5. **Validate triples**
   - Run lexical checks.
   - Run NLI on source sentence and optional context window.
   - Trigger retrieval for uncertain or weakly supported cases.
   - Run BM25 retrieval and retrieval-stage NLI.
   - Assign accept/reject/unverified status.

6. **Evaluate predictions**
   - Match predictions against ground truth.
   - Compute precision, recall, F1, false positives, and false negatives.
   - Run McNemar tests and bootstrap confidence intervals.

7. **Build ontology**
   - Cluster entities.
   - Induce candidate classes and axioms.
   - Verify axioms through statistical, reasoner, pitfall, and SHACL checks.
   - Add human-readable labels and comments.

---

## Evaluation Summary

The manuscript reports that PERSEUS improves precision over raw LLM extraction on CaRB while trading off recall. In the compact CaRB results table included in the manuscript:

| System | Precision | Recall | F1 |
|---|---:|---:|---:|
| Raw LLaMA | 46.8% | 75.8% | 57.8% |
| PERSEUS | 58.4% | 59.0% | 58.7% |

The reported precision change is +11.7 percentage points with a bootstrap 95% confidence interval of approximately `[+9.2, +14.2]`. The manuscript reports a McNemar test with `p = 2.0 x 10^-84`, indicating a systematic difference in error patterns.

The important interpretation is not that PERSEUS maximizes recall. It deliberately acts as a precision-oriented verification layer. This is appropriate when unsupported triples are more damaging than missed triples, such as in scientific, medical, legal, enterprise, or ontology-engineering contexts.

---

## Configuration Notes

### Model configuration

The manuscript and notebooks reference several model families:

| Component | Models referenced |
|---|---|
| Triple extraction | Llama 3 8B, Mistral, DeepSeek, ChatGPT variants |
| NLI validation | RoBERTa-Large-MNLI, BART-Large-MNLI in earlier drafts/context |
| Embeddings | SentenceTransformers, SBERT, BERT-based entity embeddings |
| OCR | EasyOCR, Tesseract, docTR, Donut, Nougat |
| Image captioning | BLIP, BLIP2, GIT, ViT-GPT-2 |
| Audio transcription | Whisper |

Use the manuscript and template notebooks as the source of truth for the specific experiment you are reproducing, because older draft README claims and newer manuscript numbers may differ.

### Hardware

The manuscript reports experiments using GPU-enabled hardware. Some notebooks can run on CPU but will be slow, especially for:

- LLM triple extraction,
- NLI verification,
- image captioning,
- OCR benchmarks,
- embedding-heavy clustering.

GPU acceleration is recommended for full benchmark runs.

---

## Limitations

This repository should not be treated as a complete production implementation without additional work.

Known limitations:

- Public files intentionally omit private prompts, model weights, corpora, and proprietary logic.
- Several notebooks contain redacted paths or placeholder functions.
- `Sample_Code_for_Hallucination.py` is a scaffold, not a functioning extraction/verification engine.
- `multimodal_conversion.py` requires concrete OCR, captioning, and ASR implementations to process images/audio.
- Some notebooks require Hugging Face authentication and access to external datasets.
- `clustering_algorithms_.ipynb` appears to contain malformed JSON and may need repair before opening normally in Jupyter.
- Reported metrics may differ across README drafts, notebook outputs, and manuscript revisions. The attached manuscript should be treated as the most current research context.

---


## Citation

If you use this work, cite the accompanying manuscript and the code (citation available in "CITATION.cff":

```bibtex
@article{dalal_perseus_2026,
  title   = {Hallucination-Aware Knowledge Graph Construction with LLMs},
  author  = {Dalal, Aryan Singh and McGinty, Hande Kucuk},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```



---

## Acknowledgments

We thank:
- Domain experts who reviewed ontology outputs and provided constructive feedback
- The open-source community for foundational tools (Stanza, Hugging Face Transformers, scikit-learn)
- CaRB benchmark maintainers for providing standardized evaluation datasets

---

## License

MIT License - see LICENSE file for details

---



## Additional Resources

- 📄 **Full Paper**: [Link to be soon updated with published version]
- 📊 **CaRB Benchmark**: [https://github.com/dair-iitd/CaRB](https://github.com/dair-iitd/CaRB)

---

**Last Updated**: November 2025  
**Version**: 1.0.0
