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

> **Update:** the headline result is now Experiment 2 at the bottom of this file. The tables in this first section are the original model and are kept for the record.

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

### Improvement pass result: criteria not met

| Model (held-out patients) | Seizures detected | False alarms / hour | Window AUPRC |
|---|---|---|---|
| First CNN (5 s smoothing), the headline result | 16 of 35 | 4.8 | 0.401 |
| First CNN, smoothing tuned on validation (15 s) | 14 of 35 | 4.2 | |
| Hard-negative CNN, smoothing tuned on validation (15 s) | 18 of 35 | 3.9 | 0.494 |

Pre-registered target: at most 2.0 false alarms per hour and at least 14 of 35 seizures. The hard-negative model met the sensitivity half (18 of 35) but not the false-alarm half (3.9 per hour), so by the decision rule **the first model remains the headline result** and this pass is reported as an unsuccessful attempt.

It did help modestly: window AUPRC rose from 0.401 to 0.494 and AUROC from 0.754 to 0.775, and it caught more seizures at a lower false-alarm rate than the first model. But the false alarms did not go away. Two patients still dominate: after mining, chb05 produced about 239 false alarms per day and chb10 about 139 (first model: 249 and 198). The false-alarm problem looks patient-specific, not a lack of hard negatives in general.

A second finding is about the validation set. The thresholds chosen on the 3 validation patients gave only 0.07 to 0.2 false alarms per hour with 100% sensitivity, which did not transfer to the test patients (3.9 to 4.2 per hour). Three validation patients were too easy a proxy. Choosing the operating point by leave-patient-out cross-validation over many more patients, or calibrating per patient, would be the next experiments. Neither was run.

## Experiment 2: judge on cross-validation, touch the test patients once (pre-registered)

Written and committed before the run. The 5 test patients were already seen in experiments 1 and 2, and experiment 2's idea came from looking at them. Iterating against them further would eventually find a good number by luck. So development now uses **patient-grouped 5-fold cross-validation over the 19 non-test patients** (163 seizures, every patient held out once, chb01 and chb21 kept together), and the test patients are scored **only if the development criterion is met**.

- **Change tested:** the first model z-scores each window by itself, which removes amplitude, and a rise above a patient's own background is exactly what a seizure does. The new input adds the same window divided by that patient's causal baseline (median RMS of the 10 minutes ending 30 s earlier). Both variants are trained in every fold and compared fairly. No model selection: fixed 12 epochs.
- **Development criterion:** the better variant detects at least 50% of the 163 development seizures at no more than 2.0 false alarms per hour, with smoothing (5, 15, 30 or 60 s) and threshold chosen on the same out-of-fold scores. That is a mild optimism, with two parameters chosen on 163 seizures.
- **Test criterion (only if the above holds):** train on all 19 development patients, apply the chosen operating point unchanged to the 5 test patients, and require at least 14 of 35 seizures at no more than 2.0 false alarms per hour.
- **Decision rule:** if the development criterion is not met, the first model stays the headline, whatever else is learned.

### Experiment 2 results: criteria met

**Development (5-fold patient-grouped cross-validation, 19 patients, every patient held out once):**

| Input | Seizures detected | False alarms / hour | Median latency |
|---|---|---|---|
| z-score only (as before) | 68 of 150 (45.3%) | 1.56 | 19 s |
| z-score + baseline-relative | **81 of 150 (54.0%)** | **0.71** | 15 s |

**Test (5 held-out patients, scored once with the operating point fixed from cross-validation: 15 s smoothing, threshold 0.69):**

| Model | Seizures detected | False alarms / hour | Median latency |
|---|---|---|---|
| First CNN | 16 of 35 | 4.8 | 24.5 s |
| Hard-negative CNN | 18 of 35 | 3.9 | 19.5 s |
| **Baseline-relative CNN** | **25 of 35 (71%)** | **0.53** | 18 s |

Per test patient (detected / seizures, false alarms per 24 h): chb03 7/7, 0.6; chb05 3/5, 51; chb10 7/7, 4.3; chb14 0/8, 1.9; chb20 8/8, 0.9.

Both pre-registered criteria were met: the development criterion (at least 50% at no more than 2.0 false alarms per hour) and the test criterion (at least 14 of 35 at no more than 2.0 per hour). The only change from the first model is the input: the same window divided by the patient's own causal baseline, which keeps the amplitude information that per-window z-scoring removes.

Things to know, stated plainly:
- **A denominator detail in the development gate.** 13 of the 40 seizures in chb12 are in recordings that could not be read (missing the 18-channel montage), so they were excluded from both training and evaluation; 150 of the 163 development seizures were evaluable. The code applied the 50% threshold to the 150 (54.0%). Against all 163 it would be 81 of 163 (49.7%), just under 50%. The test patients had no unreadable seizures (35 of 35 evaluated), and the test result passes its criterion on its own.
- **Wide uncertainty.** A bootstrap over the 19 development patients gives a 95% interval of roughly 31% to 80% for pooled sensitivity. With only 5 test patients any test number is imprecise.
- **Patient variation remains large.** Averaged per patient, development sensitivity is 70%, but 2 of 19 development patients (and 1 of 5 test patients, chb14) had no seizure detected, and about 4 of 19 produced more than 24 false alarms per day. Pooled figures are pulled up by patients with many well-detected seizures.
- **The test patients are not pristine.** They were seen in experiments 1 and 2, and the second experiment was motivated by their errors. This final scoring used a threshold fixed on cross-validation and was run once, which limits but does not remove that influence.
- One seed, one dataset, no external validation. Research prototype, not a medical device.
