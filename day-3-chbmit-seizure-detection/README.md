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
