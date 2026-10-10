# Day 3: CHB-MIT seizure detection from raw EEG

Seizure detection on the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/): 23 cases from 22 pediatric subjects, 256 Hz, 198 annotated seizures. (Case chb21 is the same subject as chb01, recorded 1.5 years later; the code keeps them on the same side of every split.)

## Design

- **Input:** 18 bipolar channels common to all cases, band-passed 0.5-45 Hz (zero-phase, offline), 4 s windows with a 1 s hop, z-scored per window and channel.
- **Split by patient.** Training, validation and test cases are disjoint subjects, fixed in `src/data.py` before any training.
  - Test: chb03, chb05, chb10, chb14, chb20 (35 seizures), picked for adequate seizure counts and a spread of difficulty.
  - Validation: chb08, chb18, chb23. Train: the remaining cases.
- **Models:** a small 1D ResNet on the raw windows, and a baseline of band-power features with gradient boosting.
- **Training windows:** every seizure window plus one background window per minute, excluding 60 s around each seizure.
- **Evaluation on continuous recordings.** Held-out recordings are scored end to end, one probability per second, with no sampling. Alarms are raised when a 5 s trailing average crosses a threshold, with a 5 minute refractory period. Alarms are stamped at the end of the window, so detection is causal.
  - A seizure counts as detected if an alarm falls between onset and offset + 30 s; any other alarm is a false alarm.
  - The threshold is chosen on the validation patients (highest sensitivity with at most 1 false alarm per hour) and then applied unchanged to the test patients.
- **Metrics:** seizure-level sensitivity, false alarms per hour, detection latency, per-patient results, and a sensitivity versus false-alarm curve.
- **Leakage comparison:** the same models trained on a random window-level split, to show how much it flatters window-level scores.

## Run

Colab (T4): open `train_colab.ipynb` and run the cells in order (about 42 GB download, roughly 1.5 to 3 hours). Locally, fetch the EDF files and summaries into `data/chbmit/` and run, from `src/`: `python cache.py && python train.py && python baseline.py && python score.py && python evaluate.py`. Set `SPLIT=random` for the leaky comparison.

Research prototype, not a medical device. With a few seizures per patient, confidence intervals are wide.

## Results (single run, seed 42; 5 held-out patients, 35 seizures, about 180 h)

Window-level (every seizure window plus sparse background windows):

| Model | Split | AUROC | AUPRC | Seizure prevalence |
|---|---|---|---|---|
| 1D CNN | held-out patients | 0.754 | 0.401 | 14.8% |
| Band-power + gradient boosting | held-out patients | 0.850 | 0.673 | 14.8% |
| 1D CNN | random windows (leaky) | 0.997 | 0.987 | 16.2% |
| Band-power + gradient boosting | random windows (leaky) | 0.995 | 0.982 | 16.2% |

Event-level on continuous held-out recordings (threshold set on validation patients for at most 1 false alarm per hour, then applied unchanged):

| Model | Seizures detected | False alarms / hour | Median latency |
|---|---|---|---|
| 1D CNN | 16 of 35 (46%) | 4.8 | 24.5 s |
| Baseline | 13 of 35 (37%) | 3.9 | 14.0 s |

CNN per held-out patient (detected / seizures, false alarms per 24 h): chb03 5/7, 1; chb05 1/5, 249; chb10 2/7, 198; chb14 0/8, 32; chb20 8/8, 9.

What this shows, and what it does not:
- A random window split makes both models look near-perfect (AUROC 0.995 to 0.997). Held out by patient, the CNN drops to 0.754. The leak is the main finding.
- This is not a deployable detector. The validation threshold did not transfer: the validation patients allowed at most 1 false alarm per hour, but the test patients saw 4.8. About 95% of the CNN's false alarms come from two patients (chb05, chb10).
- Performance varies enormously by patient: the CNN caught all 8 seizures in chb20 and none in chb14. With 5 test patients, any pooled number has very wide uncertainty.
- The hand-crafted baseline beat the CNN at window level (AUROC 0.850 vs 0.754), and the CNN caught more seizures at event level (16 vs 13) at a similar false-alarm rate. Neither model clearly dominates.
- Likely next steps: hard-negative mining (background is sampled at only one window per minute in training), per-patient normalisation or calibration, and more patients in training.
- One seed and one split; no confidence intervals.

## Improvement pass (pre-registered)

Written and committed before the run, so the target cannot move after seeing results.

- **What changes:** hard-negative mining. The first model is run over the *training* recordings and its worst false positives (probability at least 0.5, at least 10 s apart, at most 25 per recording, never within 60 s of a seizure) are added to training as extra negatives. A second CNN is trained from scratch with them. Validation and test patients are never mined. The smoothing window (5, 15, 30 or 60 s) and the alarm threshold are chosen on the validation patients only.
- **Success criteria, both required, on the same 5 held-out patients at the validation-selected operating point:** at most 2.0 false alarms per hour, and at least 40% of the 35 seizures detected (14 or more). For reference, the first model got 16 of 35 at 4.8 false alarms per hour.
- **Decision rule:** if both criteria hold, the improved model becomes the headline result. If not, the first model stays the headline and the improvement pass is reported as an unsuccessful attempt.
- **A caveat to keep in mind:** this idea was motivated by where the first model failed on these same test patients (false alarms concentrated in chb05 and chb10), so the second evaluation is less pristine than the first. To separate the effect of mining from post-processing, the first model is also re-evaluated with the same smoothing search.
