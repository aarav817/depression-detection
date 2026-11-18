Build one local Streamlit page with this flow: upload a video → click Analyze → display the saved model's result. No charts, sample selectors, training controls, or model explanations on the main screen.

1. **Get OpenFace working first.**

   OpenFace's `FeatureExtraction` executable converts video into the AU measurements used by this project. It is a separate installed program with model files, not simply a Python import. It was not found on the current shell's PATH when this plan was written; check whether it is installed elsewhere.

   Install/build it using its [Ubuntu/Unix instructions](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation) or [Mac instructions](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Mac-Installation). These guides are older, so the build may need adjustments for the actual computer. This is the main setup dependency and may take longer than the frontend itself.

   Run a small MP4 containing one clearly visible face through it before connecting the UI:

   ```bash
   /absolute/path/to/FeatureExtraction -f /absolute/path/to/input.mp4 -out_dir /absolute/path/to/output -aus
   ```

   Use the executable's working installation directory if it needs relative model paths. Confirm it creates a readable CSV with `success` and the 17 AU intensity columns expected in `src/dataset.py`. Record the executable path and working directory that succeeded. Match extraction settings to the dataset's documented settings where available; do not assume matching column names alone establishes identical measurements.

   **Checkpoint:** one real video produces a compatible CSV. [OpenFace command reference](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments).

2. **Install Streamlit in the existing project environment.**

   ```bash
   source .venv-research/bin/activate
   python -m pip install streamlit
   ```

   Create `requirements-frontend.txt` containing:

   ```text
   -r requirements.txt
   streamlit
   ```

   The existing project dependencies already provide pandas and joblib. Use compatible versions for loading the saved scikit-learn artifact. The first version should use `models/baseline/logistic_regression.joblib`; no GRU integration is required.

3. **Create a single `app.py` in the repository root.**

   Start with a title, a video uploader, optional preview, and an Analyze button:

   ```python
   import streamlit as st

   st.set_page_config(page_title="Facial Video Analysis", layout="centered")
   st.title("Facial Video Analysis")
   uploaded = st.file_uploader("Upload a video", type=["mp4"])
   if uploaded is not None:
       st.video(uploaded.getvalue())
   analyze = st.button("Analyze video", disabled=uploaded is None)
   ```

   Start with MP4 videos, preferably H.264 for browser playback. A file extension filter does not guarantee valid or decodable video; handle extraction failures below. Streamlit provides both the [uploader](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader) and [video preview](https://docs.streamlit.io/develop/api-reference/media/st.video).

4. **Add one helper function for the analysis.**

   Keep it in `app.py`: `analyze_video(video_bytes)`.

   Inside the function, create a `tempfile.TemporaryDirectory`, write the upload to a fixed filename such as `input.mp4`, and create an output folder. Run OpenFace through `subprocess.run()` using an argument list, `check=True`, captured output, and a timeout. Do not build a shell command from the uploaded filename. Use absolute paths and the working directory verified in step 1.

   For example, the arguments are:

   ```python
   [openface_bin, "-f", str(video_path), "-out_dir", str(output_dir), "-aus"]
   ```

   Start with a 120-second processing timeout and a 50 MB upload limit; these are demo resource limits, not evidence that a video is scientifically suitable. Check that the expected CSV was actually created. Finish reading/extracting it inside the temporary-directory block so it is not deleted too early. The directory then cleans up automatically.

5. **Reuse the existing feature extraction and saved model.**

   After OpenFace creates the CSV, use:

   ```python
   import joblib
   import pandas as pd
   from src.dataset import summarize_visual

   artifact = joblib.load(model_path)
   features, quality = summarize_visual(csv_path)
   ```

   If extraction is excluded or fewer than two successfully tracked frames exist, return an understandable error and no model result. Preserve any other quality warnings. Build the model input in its saved column order:

   ```python
   inputs = pd.DataFrame([features])[artifact["feature_columns"]]
   score = float(artifact["pipeline"].predict_proba(inputs)[0, 1])
   threshold = float(artifact["threshold"])
   prediction = int(score >= threshold)
   ```

   `model_path` should be resolved relative to `Path(__file__).resolve().parent`. Do not train, refit preprocessing, or pick a new threshold. Return the score, predicted dataset label, successful-frame count, elapsed time, and any warnings. There are no per-feature depression probabilities in this model.

6. **Connect the button and show results.**

   When Analyze is clicked, clear the old result, show `st.spinner("Analyzing video...")`, and call the helper. Display three or four small cards: predicted dataset label, model score, successfully tracked frames, and optionally processing time.

   Format the score as `0.62 / 1.00`, not as a verified personal probability of depression. Use one small caption: `Experimental student model; newly recorded videos have not been independently validated.` No charts or technical panels are needed.

   Store the result in `st.session_state` so it survives reruns. Identify the upload by a hash of its bytes; clear stored results and errors when the upload changes or is removed. Keep only completed numeric results in state, not paths to deleted temporary files.

   Handle missing OpenFace/model files, an unreadable video, a timeout, missing output columns, and failed face tracking with short messages. An error must not leave a previous video's prediction visible. Never substitute invented output when extraction fails.

7. **Apply a minimal theme and launch.**

   Create `.streamlit/config.toml`:

   ```toml
   [theme]
   base = "light"
   primaryColor = "#0F766E"
   backgroundColor = "#F6F7FB"
   secondaryBackgroundColor = "#FFFFFF"
   textColor = "#172033"
   font = "sans serif"

   [server]
   maxUploadSize = 50
   ```

   Launch from the project root:

   ```bash
   python -m streamlit run app.py
   ```

   Open the local URL printed in the terminal. The terminal must remain running. Ctrl+C stops the app. Native Streamlit widgets and this theme are enough for the first version.

8. **Check the complete flow once it works.**

   Upload a valid video and confirm that the same OpenFace CSV gives the same score in the app and plain Python. Upload a second video and verify that the first result clears. Try a video without a visible face and an invalid file; both should show a useful error. Confirm Analyze is disabled without a file and no interaction retrains the model.

   The final implementation needs only `app.py`, `.streamlit/config.toml`, and `requirements-frontend.txt`, plus the locally installed OpenFace executable and existing model. Keep all temporary uploads and extracted files out of Git.
