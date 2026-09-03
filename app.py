"""
Streamlit front end for the Retail OOH Siting Engine.

Run locally with:  streamlit run app.py
Deploy for free at: share.streamlit.io (see deployment steps in chat)
"""

import streamlit as st
import streamlit.components.v1 as components

from engine import run_store, build_store_map, build_store_onepager

st.set_page_config(page_title="Retail OOH Siting Engine", layout="wide")

st.title("Retail OOH Siting Engine")
st.caption(
    "Give it a store location and an OOH package, get back where to place "
    "each unit and why -- built from public OpenStreetMap data."
)

with st.form("store_form"):
    col1, col2 = st.columns(2)

    with col1:
        city = st.text_input("Store name / city", "")
        latitude = st.number_input("Latitude", value=22.5726, format="%.6f")
        longitude = st.number_input("Longitude", value=88.3639, format="%.6f")
        radius_m = st.number_input(
            "Search radius (metres)", value=3000, min_value=500, step=500
        )

    with col2:
        auto_tops = st.number_input("Auto Tops", min_value=0, value=75)
        pole_kiosks = st.number_input("Pole Kiosks", min_value=0, value=150)
        no_parking_boards = st.number_input(
            "No Parking Boards", min_value=0, value=400
        )

    submitted = st.form_submit_button("Run siting analysis")

if submitted:

    if not city.strip():
        st.error("Enter a store name before running.")
        st.stop()

    with st.spinner(
        "Pulling OpenStreetMap data and running the engine -- "
        "this can take anywhere from a few seconds to a couple of "
        "minutes depending on how dense the area is..."
    ):
        try:
            result = run_store(
                city=city,
                latitude=latitude,
                longitude=longitude,
                auto_tops=int(auto_tops),
                pole_kiosks=int(pole_kiosks),
                no_parking_boards=int(no_parking_boards),
                radius_m=int(radius_m),
                return_full_result=True
            )
        except Exception as exc:
            st.error(f"The run failed: {exc}")
            st.stop()

    report = result["management_report"]
    counts = report["location_counts"]

    st.success(
        f"Done -- {counts['auto_tops']} corridors, "
        f"{counts['pole_kiosks']} kiosk clusters, "
        f"{counts['no_parking_boards']} board sites recommended."
    )

    tab_auto, tab_kiosk, tab_board, tab_map = st.tabs(
        ["Auto Tops", "Pole Kiosks", "No Parking Boards", "Map"]
    )

    with tab_auto:
        st.dataframe(
            report["recommendations"]["auto_tops"], use_container_width=True
        )

    with tab_kiosk:
        st.dataframe(
            report["recommendations"]["pole_kiosks"], use_container_width=True
        )

    with tab_board:
        st.caption(
            f"Placement method: {report['no_parking_boards_placement_method']}"
        )
        st.dataframe(
            report["recommendations"]["no_parking_boards"],
            use_container_width=True
        )

    with tab_map:
        store_map = build_store_map(result["full_result"])
        components.html(store_map._repr_html_(), height=650)

    st.divider()

    onepager_html = build_store_onepager(result)
    st.download_button(
        "Download one-pager (HTML)",
        data=onepager_html,
        file_name=f"{city.replace(' ', '_')}_siting_onepager.html",
        mime="text/html"
    )
