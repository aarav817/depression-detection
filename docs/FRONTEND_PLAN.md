This is a simple frontend plan for your December 2, 2026 presentation. One teammate finishes the model; the other builds the interface using this repository. The frontend agent already has the codebase and should inspect it directly.

1. **Build one Streamlit page.**

   The demo should let you select a participant example, see a facial-movement chart, click Analyze, and compare the prediction with the dataset label. Use the current logistic-regression model immediately. Add the final model only when the model teammate has a working way to load it and predict.

   Use Python and Streamlit. No JavaScript, separate API, database, login, deployment service, or webcam feature is needed. Run the app locally on the presentation laptop.

   **Finished product:** one attractive page that demonstrates a real prediction reliably in a 5–7 minute presentation.

2. **Divide the work between the two teammates.**

   | Model teammate | Frontend teammate |
   |---|---|
   | Finish training and testing | Build the Streamlit page |
   | Save the final model and preprocessing | Load and use the saved model |
   | Confirm how to make one prediction | Show the prediction clearly |
   | Supply the final performance numbers | Display those numbers with the correct validation/test label |

   The frontend teammate can start now with the existing baseline. Both teammates should agree before editing the same source file. Keep frontend work on a `frontend-demo` Git branch until it works.

3. **Have the frontend agent inspect the existing code.**

   Tell the agent to read these files before writing the app:

   | Existing file | What the frontend can reuse |
   |---|---|
   | `src/dataset.py` | `summarize_visual()` converts one visual CSV into the 68 baseline features and a quality report |
   | `src/train.py` | Shows how the saved model is structured and how scores become labels |
   | `data/processed/manifest.csv` | Links participants, labels, and visual CSV paths |
   | `data/processed/splits.csv` | Identifies validation examples |
   | `models/baseline/logistic_regression.joblib` | The trained baseline and its preprocessing, feature order, and threshold |
   | `results/baseline/validation_metrics.json` | Actual validation results for the performance panel |

   The agent should check which data and model files are present on its machine. They are ignored by Git, so having the source repository alone does not guarantee these local files exist. If needed, the model teammate can share the saved model and a few example CSVs. There is no need to redownload the entire dataset for the frontend.

   **Checkpoint:** identify an available model and at least one example CSV before connecting predictions.

4. **Set up Streamlit and make a page appear.**

   Open the project in VS Code, open its terminal, and activate the existing environment:

   ```bash
   source .venv-research/bin/activate
   python -m pip install streamlit
   ```

   Create `requirements-frontend.txt`:

   ```text
   -r requirements.txt
   streamlit
   ```

   On a machine that still needs the project packages, use:

   ```bash
   python -m pip install -r requirements-frontend.txt
   ```

   Create `app.py` in the project root:

   ```python
   import streamlit as st

   st.set_page_config(page_title="Facial Movement Classifier", layout="wide")
   st.title("Facial Movement Classifier")
   st.caption("A student project using facial-movement measurements.")
   st.info("Choose an example and run the model.")
   ```

   Start it from the project directory:

   ```bash
   python -m streamlit run app.py
   ```

   Open the local address printed in the terminal, normally `http://localhost:8501`. Keep the terminal running. Press Ctrl+C to stop the app. These commands work on Ubuntu/WSL and Mac with the environment installed.

   **Checkpoint:** the page opens in your browser. Do not move on until this works.

5. **Make the page look clean.**

   Use this layout:

   ```text
   Facial Movement Classifier
   A short explanation of what the project does.

   Choose example: [Sample A v]       Facial movements over time
   [ Analyze sample ]                [simple line chart]

   [ Predicted class ]   [ Model score ]   [ Valid frames ]

   [ Reveal dataset label ]

   Model performance
   [Small comparison table]

   [How it works — expandable]
   ```

   Use `st.columns()` for the input/chart layout and result cards, `st.selectbox()` for samples, `st.button()` for actions, `st.metric()` for numerical results, and `st.expander()` for details. Start with one model; a model selector is optional if both models are ready.

   Create `.streamlit/config.toml`:

   ```toml
   [theme]
   base = "light"
   primaryColor = "#0F766E"
   backgroundColor = "#F6F7FB"
   secondaryBackgroundColor = "#FFFFFF"
   textColor = "#172033"
   font = "sans serif"
   ```

   Keep the interface mostly white, with dark text and one teal accent. Use clear labels and generous spacing. Native Streamlit components and this theme are enough; custom CSS is optional. Use a system font so the page works without internet access.

   **Checkpoint:** the page is readable at the resolution you will present at and does not feel crowded.

6. **Add prepared examples and a movement chart.**

   Use three or four validation examples from the existing manifest and splits. Name them Sample A, Sample B, and Sample C in the interface so the menu does not reveal the answer. Include an example the model gets wrong for discussion. Selected examples illustrate behavior; aggregate metrics describe performance.

   When someone chooses an example, load its visual CSV with pandas and strip whitespace from column names. Use `st.line_chart()` to show a few AU intensity columns, such as AU01_r, AU12_r, and AU45_r. Use timestamp for the horizontal axis if it is valid; otherwise use frame number. Label the chart so the audience knows these are facial-movement measurements.

   You do not need a face image or animation. The chart is a useful visual that directly reflects the model's input.

   **Checkpoint:** selecting another example changes the chart and clears any previous prediction.

7. **Connect the Analyze button to the real model.**

   Reuse the current code instead of creating another backend. For the existing baseline, the logic is:

   ```python
   from pathlib import Path
   import joblib
   import pandas as pd
   from src.dataset import summarize_visual

   project = Path(__file__).resolve().parent
   artifact = joblib.load(project / "models/baseline/logistic_regression.joblib")
   features, quality = summarize_visual(selected_csv_path)

   if features is None:
       st.error("This sample does not contain enough usable facial measurements.")
   else:
       inputs = pd.DataFrame([features])[artifact["feature_columns"]]
       score = float(artifact["pipeline"].predict_proba(inputs)[0, 1])
       threshold = float(artifact["threshold"])
       predicted_label = int(score >= threshold)
   ```

   This is the prediction logic to integrate into the button handler, not a complete standalone app. Resolve the selected CSV path from the project root as well.

   The saved pipeline already contains preprocessing. Do not fit anything, retrain a model, or change its threshold in the app. Do not pass the dataset label into the prediction input. Show any quality warnings returned by the existing feature extractor.

   Display the predicted dataset class, a score such as `0.62 / 1.00`, and the usable-frame count. Explain that the score is a model output, not a medical diagnosis or a verified probability of depression.

   Cache the loaded model if useful, and restart the app when replacing model artifacts. Use `st.session_state` to keep the result visible after another button click. Clear the result when the example or model changes. A failed analysis must not leave an old prediction on screen.

   Add Reveal dataset label after analysis, using the label from the manifest. Display whether the prediction matches it. Compare the app's output with the same sample run directly in Python to confirm the integration is correct.

   **Checkpoint:** the app makes a genuine prediction and displays the correct corresponding label when requested.

8. **Add a small model-performance panel.**

   Read the existing results JSON and show dummy versus logistic regression using balanced accuracy, recall, and precision. Label the current numbers as Validation results. When the model teammate provides final test results, display those as Test results instead. Do not relabel validation results as test results or compare models evaluated on different groups without explaining that difference.

   Use one short expandable explanation for what AUs are, how the model uses them, and one limitation. Avoid filling the screen with every metric or technical setting.

   If the final GRU is ready, the model teammate supplies its checkpoint and working prediction code, including sequence preprocessing. The frontend teammate connects that code in place of or alongside the baseline. The 68-summary-feature path above is not the GRU input path. Do not delay a working demo for an unfinished GRU.

   **Checkpoint:** the audience can understand what went in, what came out, and how the model performs overall.

9. **Check the demo manually and fix obvious failures.**

   Verify these cases:

   - The app starts from a fresh terminal using the documented command.
   - Every prepared example loads and produces the expected score/class.
   - Clicking Reveal does not make the prediction disappear.
   - Switching examples clears the old prediction and hidden-label state.
   - Missing model/data files produce a useful message instead of a broken screen.
   - The chart and result cards are readable on the presentation display.
   - The app works without internet after installation and file transfer.
   - No click starts model training or final test evaluation.

   Run the existing test suite after integration:

   ```bash
   python -m unittest discover -s tests -v
   ```

   A few frontend behavior tests are useful if easy to add, but do not build an elaborate testing framework for the demo. Run the app on both teammates' computers and rehearse on the actual presentation laptop.

   **Checkpoint:** one teammate can demonstrate the project without the other operating a second terminal.

10. **Finish before the deadline and rehearse.**

   | Deadline | Deliverable |
   |---|---|
   | Nov 20 | Page opens and has the basic theme/layout |
   | Nov 23 | Prepared examples, chart, and baseline prediction work |
   | Nov 27 | Final model, if ready, is connected; performance panel is complete |
   | Nov 30 | Stop adding features; check the presentation laptop and record a backup |
   | Dec 1 | Rehearse the full presentation twice |
   | Dec 2 | Present |

   A six-minute presentation can spend one minute on the project, one minute on the data/model, two minutes demonstrating a correct and incorrect prediction, one minute on overall results, and one minute on limitations and teamwork.

   Keep a short screen recording of a real working run as a fallback. If used, say it is recorded. Start the app and run one example before the presentation begins.

   The required frontend files are just `app.py`, `.streamlit/config.toml`, and `requirements-frontend.txt`. Add a short startup note to the README once it works. CSV upload is an optional extra only after the prepared-example demo is reliable. There is no need for more features to deliver a good student presentation.

Useful references: [Streamlit installation](https://docs.streamlit.io/get-started/installation/command-line), [theming](https://docs.streamlit.io/develop/concepts/configuration/theming), and [session state](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state).
