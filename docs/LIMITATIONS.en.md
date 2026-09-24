# Known Limitations and Improvement Directions

*English · [中文](LIMITATIONS.md)*

This document collects the system's current capability boundaries and known defects; each entry states what it affects, and the directions available for improvement.

---

## 1. Measurement Validity

### 1.1 Vehicle counts are not traffic flow

The system recognises vehicles in single frames and performs no cross-frame tracking. The same vehicle is counted again in consecutive frames, so the output represents "the number of vehicles visible in a frame at a given moment", not the flow passing a cross-section.

The discrepancy widens under congestion: vehicles move slowly, so a larger share of frames contains the same vehicle. Every downstream metric and every forecasting target inherits this definition.

**Improvement direction**: introduce object tracking or multi-frame association; or define a virtual detection line and count crossings, which yields flow in the proper sense.

### 1.2 Detection accuracy has not been validated against manual annotation

The detector uses public pretrained weights. No samples from this project's imagery have been manually annotated, so measured precision, recall and mAP are not available.

The counting network's validation metric is its disagreement with the detector's counts, and the detector itself is unvalidated. That metric therefore measures agreement between two models, not absolute accuracy.

**Impact**: models can be compared with each other, but no claim can be made about "how accurate the recognition is".

**Improvement direction**: manually annotate a small representative set (a few hundred images is sufficient) and report the detector's P/R/mAP together with vehicle-count error.

### 1.3 Measurement fails at night

At night, detection recall and confidence both fall sharply. For the two cameras that perform worst after dark, almost every frame in the early-morning hours is marked unmeasurable by the measurability gate.

**Impact**: congestion assessment is only valid under measurable conditions, and night-time frames do not contribute to the band statistics. This is a deliberate gate — an unmeasurable frame must not be presented as free-flowing — but it means the system has no all-weather capability.

**Improvement direction**: preprocess night imagery (exposure normalisation, denoising); or add night-time samples for detector fine-tuning.

### 1.4 Accuracy limits of the road masks

Masks are derived from detection-box coverage statistics, and their accuracy is limited in two ways:

- For cameras with sparse detection coverage at night, the mask is conservative and may omit part of the real carriageway;
- For wide, high-angle mounts, the mask is loose and takes in shoulders and planted verges.

**Impact**: mask error is a constant offset per camera. It does not change the time-series ordering within a camera (which is the dimension both congestion assessment and forecasting operate on), but it does affect density comparisons between cameras.

**Improvement direction**: correct mask boundaries manually; or refine boundaries using image information rather than detection boxes alone.

---

## 2. Data Quality

### 2.1 Missing records imputed as zero

After time alignment, any camera–time step without a record is filled with zero and flagged. The two batches are roughly 24% and 25% imputed.

**Impact**: imputing zero equates "missing data" with "no vehicles". When a camera is unobserved for a sustained period, the model learns a false stretch of low flow. This is most pronounced at night.

**Improvement direction**: use a distinguishable missing marker and let the model handle it explicitly; or down-weight imputed samples during training.

### 2.2 Duplicate frames

The same view may be collected at several time points when a camera has not refreshed. Earlier data contains several hundred such frames.

**Impact**: identical content appearing at different times slightly distorts time-based distributions and percentile thresholds.

**Improvement direction**: de-duplicate by content hash, or merge them into a single observation in the time series.

### 2.3 A step introduced by a change in collection policy

The more recent batch had its frame-freshness threshold changed during collection, relaxed from 15 minutes to 240 minutes.

**Impact**: before the change, early-morning frames were systematically discarded as stale; afterwards, imagery up to four hours old is treated as current. This creates a step in the time series that is unrelated to real traffic. Any apparent surge near that timestamp should be checked against this cause first.

**Improvement direction**: exclude the affected window at the feature or sample level; or mark it explicitly in the data.

### 2.4 The two batches are not directly comparable

The batches differ in both sampling interval (5 vs 10 minutes) and camera set.

**Impact**: "forecast step 12" means 60 minutes in the first batch and 120 minutes in the second. Comparing their errors directly confounds data differences with model differences.

**Improvement direction**: align the sampling interval and forecast horizon before comparing; or state explicitly the conditions under which the comparison holds.

---

## 3. Modelling and Evaluation

### 3.1 Single data split

Data is split once, chronologically, into 60% / 20% / 20%, and evaluated on one test interval.

**Impact**: a single result is sensitive to where the split falls and does not establish stable performance.

**Improvement direction**: use rolling time windows or multiple test intervals, and report the distribution of metrics rather than a single value.

### 3.2 Limited data volume

The available data covers 8 cameras over roughly one week each, giving test sets in the hundreds of samples.

**Impact**: both model capacity and hyperparameter choice are constrained by this; generalisation cannot be established from data of this scale.

**Improvement direction**: extend the collection span and include more cameras; larger models are not warranted before the data volume grows.

### 3.3 The congestion band is a relative measure

The band is determined by the density percentile within the same camera and hour of day. By construction, every camera necessarily has some proportion of frames classified as congested in every hour.

**Impact**: the band expresses relative position, and cannot be reported as "the road is congested X% of the time". Density is likewise relative: without calibrated lane counts and lane geometry it cannot be converted into an absolute unit such as vehicles per square kilometre.

**Improvement direction**: calibrate lane counts and geometry and define thresholds on absolute density; this requires lane-level information beyond the mask.

### 3.4 The spatial graph is hand-configured

The camera connections used by the spatio-temporal graph model come from a manually defined regional grouping. They are neither learned from data nor an accurate road topology.

**Impact**: the model's graph structure does not reflect real traffic propagation, and its results should not be read as evidence of spatial dependence.

**Improvement direction**: build the graph from validated geographic distance or road connectivity, and compare against a model without a spatial graph.

### 3.5 No hyperparameter search

Each model is trained once with a single fixed configuration.

**Impact**: comparisons between models reflect relative performance under default settings, not the capability ceiling of each.

**Improvement direction**: once individual models are stable, introduce a systematic search with a fixed budget so results remain comparable.

### 3.6 No external features

Current features are vehicle counts and time encodings (hour of day, day of week). Weather, public holidays and incidents are not included.

**Improvement direction**: introduce them incrementally, once they can be obtained reliably and validated.

---

## 4. Engineering and Reproducibility

### 4.1 Data and weights are not distributed with the repository

The image datasets and model weights are excluded from version control for size reasons.

**Impact**: a fresh clone cannot run the full pipeline directly; the data must be obtained separately and the weights trained or acquired as documented.

**Improvement direction**: package them as a separate archive on delivery, or provide deterministic acquisition and generation steps in the documentation.

### 4.2 Dependencies are not locked to full reproducibility

Direct dependencies and their versions are recorded, but no lock file covering all transitive dependencies is provided.

**Impact**: environments rebuilt at a different time or on a different machine may resolve slightly different dependency sets.

**Improvement direction**: add a lock file and record the runtime device and software versions.

### 4.3 No automated tests

The project contains no unit or integration tests.

**Impact**: changes to the pipeline have no automated correctness protection.

**Improvement direction**: add tests for deterministic stages such as data conversion and metric computation.

---

## 5. Summary of Improvement Directions

Ordered by their effect on the credibility of the conclusions:

**Priority 1 — determines whether the conclusions hold**

1. Manually annotate a small set of images and report the detector's precision, recall and counting error, giving the perception stage a citable accuracy figure.
2. Introduce cross-frame tracking or a virtual detection line, upgrading the output from "vehicles visible in a frame" to traffic flow.
3. Add night-time samples or night imagery preprocessing, widening the measurable window.

**Priority 2 — determines whether the experiments are credible**

4. Use rolling time windows or multiple test intervals, and report the distribution of metrics.
5. Extend the data volume (time span and camera count).
6. Calibrate lane counts and geometry so congestion thresholds can be defined on absolute quantities.

**Priority 3 — determines whether the models still have headroom**

7. Rebuild the spatial graph from geographic distance or road connectivity, and compare against a model without a spatial graph.
8. Introduce a systematic hyperparameter search once the data volume grows.
9. Consider ensemble methods only after individual models and baselines are stable.
