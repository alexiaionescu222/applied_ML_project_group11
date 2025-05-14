# run_pipeline.py
import streamlit as st
import subprocess
import sys

# List of pipeline steps
pipeline_steps = [
    "split.py",
    "preprocessing.py",
    "test_preprocessing.py",
    "train_baseline_knn.py",
    "train_cnn.py",
    "evaluate_test.py"
]

def run_script(script_name):
    """Runs a Python script using the current Python interpreter."""
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error running {script_name}:\n{e.stderr or e}"

# Streamlit app UI
st.title("Music Genere Classification")

st.markdown("Select a step to run or click **Run All** to execute the full pipeline.")

if st.button("Run All Steps!"):
    for step in pipeline_steps:
        st.subheader(f"Running: `{step}`")
        output = run_script(step)
        st.code(output)
    st.success("All steps completed successfully!")

st.markdown("---")

selected_step = st.selectbox("Run individual step:", pipeline_steps)

if st.button("Run Selected Step"):
    st.subheader(f"Running: `{selected_step}`")
    output = run_script(selected_step)
    st.code(output)
