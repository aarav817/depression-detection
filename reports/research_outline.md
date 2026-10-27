# Research report outline

## Research question
Can existing facial behavior measurements predict AVDD-TCMI participant labels on unseen participants? Does a temporal GRU improve on summary-feature logistic regression?

## Dataset and provenance
- Source: Mar-rill/AVDD-TCMI-dataset on Hugging Face.
- Record the exact revision, download date, license/access terms, and label definitions from verified documentation.
- Describe HDS and SDS collection settings separately; verify participant counts and class proportions from the released labels.
- Document feature meanings, tracking quality, recordings per participant, exclusions, and missingness.
- Check for repeated identities across recordings/subsets and duplicate recordings.

## Methods
- Restrict primary inputs to verified facial measurements. Retain subset/gender only as evaluation metadata.
- Save approximately 70/15/15 participant-level train/validation/test assignments, stratifying by subset and label with seed 42.
- Summarize valid action-unit intensities using mean, standard deviation, median, and IQR. Average recording vectors per participant.
- Compare a most-frequent dummy with median-imputed, standardized, balanced logistic regression; select C using validation average precision.
- GRU proposal: up to 256 evenly sampled valid observations, hidden size 64, dropout 0.2, training-weighted binary logit loss, Adam 0.001, batch 16, maximum 50 epochs, patience 7. Validate implementation and record final settings before running.
- Fit every preprocessing transform using training participants only. Aggregate recording scores to participant scores.

## Evaluation protocol
Select thresholds using validation balanced accuracy; freeze preprocessing, model, and threshold before testing. Report participant counts, precision, recall, specificity, balanced accuracy, average precision, ROC-AUC, and confusion matrices. Report HDS/SDS results separately and gender groups where sample sizes support interpretation. Add participant bootstrap confidence intervals and precision–recall plots when implemented. Treat cross-setting transfer as separate experiments with independent tuning rules.

## Results

| Model | Test participants | Precision | Recall | Specificity | Balanced accuracy | AP | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dummy | | | | | | | |
| Logistic regression | | | | | | | |
| GRU | | | | | | | |

Group results, confidence intervals, plots, and error analysis:

## Limitations
Facial behavior alone cannot establish a clinical diagnosis. Dataset labels may reflect collection and annotation practices. Class imbalance, setting differences, tracking failures, demographic representation, identity leakage, and limited cohort size can affect conclusions. Scores are not calibrated clinical probabilities. Evenly sampling recordings changes temporal resolution and does not preserve absolute movement speed. Generalization to webcam recordings or other populations is untested. Respect dataset terms, consent, and participant privacy.

## Reproducibility
Record dataset revision, exclusion log, split assignments, feature order and extraction settings, dependency versions, seeds, model settings, threshold, checkpoints, and evaluation code version. Do not publish identifiable participant records.
