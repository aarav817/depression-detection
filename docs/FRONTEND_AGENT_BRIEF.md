Give the frontend agent this instruction while it is opened in the repository:

```text
Build a simple, polished Streamlit frontend for our student project presentation
on December 2. Read docs/FRONTEND_PLAN.md and inspect the existing code and local
model/data files first. You already have the whole repository; reuse its code.

Start with the saved logistic-regression model and summarize_visual() from
src/dataset.py. Make one page with a prepared-example selector, a facial-movement
line chart, an Analyze button, prediction/score/valid-frame cards, a separate
Reveal dataset label button, and a small model-performance table.

Use a light theme, white panels, dark text, and a teal accent. Native Streamlit
components are enough. Keep the implementation primarily in app.py, with
.streamlit/config.toml and requirements-frontend.txt. Do not add a separate API,
React app, database, login, webcam feature, or elaborate architecture.

Keep the true dataset label separate from prediction inputs. Reuse saved
preprocessing, feature order, and threshold. Do not train or tune in the app.
Use session state so results persist when revealing labels, and clear stale
results when inputs change. Display real model outputs; do not invent scores.

The other teammate owns training/testing and will supply the final model's
working prediction code. Use the baseline now and connect the final model later
if ready. Do not rewrite their training code. If ignored model/data files are
missing, identify the exact files needed from the teammate.

Work step by step: get a page running, style it, load examples/chart, connect
real prediction, show results, then check the presentation workflow. Explain
commands simply because we are new to frontend development. Preserve existing
work and use a frontend-demo branch. Do not push or merge without authorization.

At the end, give us the startup command and a short manual checklist. Prioritize
a clean, reliable six-minute demonstration over extra features.
```
