# tess-bls-cnn-pipeline
BLS-CNN pipeline for short-period planet detection in TESS light curves

## Pipeline overview

| Script | Purpose |
|--------|---------|
| `Phase_1.py` | Download field star backgrounds for CNN training |
| `Phase_2.py` | Build training windows via transit injection |
| `phase_3_train.py` | Train the CNN classifier |
| `phase_5_sweep.py` | Run the full sector sweep or confirmed-planet benchmark |
| `morphology_filters.py` | Automated false-positive vetting checks |
| `vet_candidates.py` | Generate folded light curve plots for visual vetting |
| `analyze_batch.py` | Classify candidates by tier after each batch |

## Requirements
pip install lightkurve astroquery tensorflow numpy pandas matplotlib astropy scipy
## Trained model

Download `final_phase3_model.keras` from the [Releases page](link).
Place it in the same directory as `phase_5_sweep.py`.