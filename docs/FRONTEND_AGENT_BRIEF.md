Use this prompt with an agent opened in the repository:

```text
Implement docs/FRONTEND_PLAN.md as a bare-bones Streamlit video demo.
Read the existing code first. Build one app.py plus a simple theme and frontend
requirements file. The UI is: upload MP4, optional video preview, Analyze button,
then predicted dataset label, model score, valid-frame count, and optional runtime.
No charts, sample menus, team schedules, separate APIs, or model explanation panels.

First locate/install and verify OpenFace FeatureExtraction on one real video.
Do not claim the upload-to-result flow works until actual extraction succeeds.
Call the executable with subprocess arguments and a timeout, using temporary
folders. Check its CSV has the required AU columns and success field.
Reuse summarize_visual() from src/dataset.py and the trusted existing artifact
models/baseline/logistic_regression.joblib. Use its saved feature order,
preprocessing, and threshold. Do not train or tune anything from the frontend.

Use a spinner, simple result cards, and session state. Clear stale predictions
when the upload changes, and handle missing dependencies, bad videos, timeouts,
and failed face tracking. Display a model score rather than a verified medical
probability. Do not invent per-feature depression probabilities or fake results.

Keep it simple. Explain the startup command and verify the actual upload,
extraction, and prediction path. Preserve existing work and do not push or merge
without authorization.
```
