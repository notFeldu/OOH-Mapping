"""
Batch Processing -- run the siting engine across many stores from
one spreadsheet, get every store's management report, vendor sheet,
and map back as one ZIP download.
"""

import pandas as pd
import streamlit as st

from engine import run_batch_activities, build_batch_zip

st.set_page_config(page_title="Batch Processing", layout="wide")

st.title("Batch Processing")
st.caption(
    "Upload a spreadsheet with one row per store. Required columns: "
    "Store Name, Lat, Long, Auto Tops, Pole Kiosks, No Parking Boards. "
    "Optional: Store Code (used as the activity ID if given), "
    "Search Radius, Budget, Go Live Date."
)

uploaded_file = st.file_uploader(
    "Store list (.xlsx or .csv)", type=["xlsx", "xls", "csv"]
)

if not uploaded_file:
    st.info("Upload a spreadsheet to get started.")
    st.stop()

try:
    if uploaded_file.name.lower().endswith(".csv"):
        stores_df = pd.read_csv(uploaded_file)
    else:
        stores_df = pd.read_excel(uploaded_file)
except Exception as exc:
    st.error(f"Couldn't read that file: {exc}")
    st.stop()

st.subheader(f"Preview -- {len(stores_df)} store(s)")
st.dataframe(stores_df, use_container_width=True)

default_radius = st.number_input(
    "Default search radius (metres) -- used for any row without its "
    "own Search Radius value",
    value=3000, min_value=500, step=500
)

run_clicked = st.button("Run batch", type="primary")

if run_clicked:

    progress_bar = st.progress(0)
    status_text = st.empty()
    log_lines = []
    log_box = st.empty()

    def update_progress(index, total, store_name, status, error=None):
        progress_bar.progress(index / total)
        status_text.text(f"[{index}/{total}] {store_name}: {status}")
        line = f"{'✅' if status == 'success' else '❌'} {store_name}"
        if error:
            line += f" -- {error}"
        log_lines.append(line)
        log_box.text("\n".join(log_lines))

    with st.spinner(
        "Running the engine store by store -- each one takes "
        "anywhere from a few seconds to a couple of minutes, so a "
        "full batch can take a while..."
    ):
        successes, failures = run_batch_activities(
            stores_df,
            search_radius=default_radius,
            progress_callback=update_progress
        )

    st.divider()

    col1, col2 = st.columns(2)
    col1.metric("Succeeded", len(successes))
    col2.metric("Failed", len(failures))

    if failures:
        st.subheader("Failures")
        st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)

    if successes:
        st.subheader("Download")
        zip_bytes = build_batch_zip(successes)
        st.download_button(
            f"Download all {len(successes)} store(s) (ZIP)",
            data=zip_bytes,
            file_name="batch_activity_outputs.zip",
            mime="application/zip"
        )
        st.caption(
            "One folder per store inside the ZIP, named by its "
            "activity ID -- each with management_report.html, "
            "vendor_sheet.csv, and map.html."
        )
    else:
        st.warning("No stores succeeded -- nothing to download.")
