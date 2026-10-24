# Sample data

| File | What it is | Source | License |
|---|---|---|---|
| `chest_xray/` | 5,856 pediatric chest X-rays (JPEG) in `train/` (5,216), `test/` (624) and `val/` (16), each split into `NORMAL/` and `PNEUMONIA/` | D. Kermany, K. Zhang and M. Goldbaum, "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification", Mendeley Data, 2018 (as packaged on Kaggle: paultimothymooney/chest-xray-pneumonia) | CC BY 4.0 |
| `telemetry_1.xlsx` | 5,644 patients tested for SARS-CoV-2 at Hospital Israelita Albert Einstein (São Paulo), anonymized: 111 columns of lab results (standardized), test result and admission to ward, semi-intensive unit or ICU. Blood counts exist for 603 patients | Kaggle: einsteindata4u/covid19, "Diagnosis of COVID-19 and its clinical spectrum" | As published on Kaggle by the hospital |
| `mtsamples.csv` | 4,999 sample medical transcriptions with specialty, description and keywords (4,966 with text) | Kaggle: tboyle10/medicaltranscriptions, scraped from mtsamples.com | CC0 1.0 on Kaggle |
| `radiology_metadata.csv` | 100 imaging study records (study type, diagnosis, slice count) | Created by the author for the first edition; synthetic | MIT (this repository) |

How the code uses them:

- **X-rays**: the pneumonia model trains on a balanced sample of `train/` (500 per class by default) and is tested on all of `test/`. The workload sends `test/` images only, and only `test/` is uploaded to Modal.
- **Labs**: the admission model uses the 14 blood counts and age quantile of the 603 patients who have them (70/30 split).
- **Notes**: the note model learns to spot Cardiovascular / Pulmonary transcriptions (80/20 split).
- **Radiology metadata**: not used by the current pipeline (the first edition encoded it with an untrained network); kept for reference.

The three datasets describe different people. A triage "case" pairs one held-out X-ray, lab panel and
note; the pairing is synthetic, the records and labels are real.
