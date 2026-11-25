# Data licenses

This file lists the datasets and models this playbook uses, where they come from, and their licenses. The code in this repository is MIT-licensed. That license does not cover third-party data or models; each one keeps its own license, listed below.

## Datasets

| Dataset | Source | License | In this repo |
|---|---|---|---|
| Chest X-ray pneumonia images (5,856 pediatric chest X-rays in `train/`, `test/` and `val/`) | Kermany, Zhang and Goldbaum, Mendeley Data, 2018, as packaged on [Kaggle: paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `sample-data/chest_xray/` (1.2 GB) |
| Diagnosis of COVID-19 and its clinical spectrum (5,644 anonymized patients, lab results) | Hospital Israelita Albert Einstein, [Kaggle: einsteindata4u/covid19](https://www.kaggle.com/datasets/einsteindata4u/covid19) | Unclear (see Notes) | `sample-data/telemetry_1.xlsx` (568 KB) |
| Medical transcriptions (4,999 sample transcriptions) | [Kaggle: tboyle10/medicaltranscriptions](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions), scraped from mtsamples.com | CC0 1.0, as listed on Kaggle. Upstream terms unclear (see Notes) | `sample-data/mtsamples.csv` (16 MB) |
| Radiology study metadata (100 records) | Synthetic, created by the author for the first edition | MIT (this repository) | `sample-data/radiology_metadata.csv` (4 KB). Not used by the current pipeline |

## Models

| Model | Source | License | How it's used |
|---|---|---|---|
| ResNet-18, ImageNet weights (`ResNet18_Weights.DEFAULT`) | [torchvision](https://pytorch.org/vision/stable/models.html) | BSD-3-Clause (torchvision), as the README states | X-ray image features (`atlas_pipeline/encoders.py`, `serverless_multimodal_pipeline.py`). Downloaded by torchvision on first use, during `make setup` |
| all-MiniLM-L6-v2 | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Apache-2.0 | Clinical note embeddings. Downloaded from Hugging Face on first use, during `make setup` |
| Qwen3-1.7B | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | Apache-2.0 | Local LLM scaling advisor (`atlas_pipeline/advisor.py`, `local_llm_scaling_demo.py`, `make smoke-full`). Downloaded from Hugging Face on first use (about 4 GB). Set by `ATLAS_ADVISOR_MODEL` |
| Qwen3-4B | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | Apache-2.0 | The same advisor on a Modal GPU (`predictive_autoscaler.py`). Downloaded into the Modal image. Set by `ATLAS_MODAL_ADVISOR_MODEL` |

## Notes

- `telemetry_1.xlsx`: the license is unclear. The repo says it is used "as published on Kaggle by the hospital" and does not name a license.
- `mtsamples.csv`: Kaggle lists it as CC0 1.0. The repo says the data was scraped from mtsamples.com and does not state that site's terms.
- The three datasets describe different people. A triage case pairs one record of each; the pairing is synthetic, the records and labels are real.

## Citations

- Chest X-rays: D. Kermany, K. Zhang and M. Goldbaum, "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification", Mendeley Data, 2018.
- ResNet: K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition", arXiv:1512.03385, 2015.
- Qwen3-1.7B and Qwen3-4B: Qwen Team, "Qwen3 Technical Report", arXiv:2505.09388, 2025.
- COVID-19 lab data, medical transcriptions, all-MiniLM-L6-v2: no citation given by the source.
