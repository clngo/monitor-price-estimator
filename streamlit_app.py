import os
import streamlit as st

from src.pipeline.predict_pipeline import PredictPipeline


st.set_page_config(page_title="Monitor Price Estimator", page_icon="🖥️", layout="centered")
st.title("Monitor Price Estimator")
st.caption("Estimate a fair price based on basic monitor specs.")
st.markdown(
    """
    <style>
    .value-accent {
        color: #9CC9FF;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

brand_options = ["Acer", "Dell", "LG", "Samsung", "ASUS", "BenQ"]


with st.form("price_form"):
    st.subheader("Core specs (required)")
    brand_choice = st.selectbox("Brand", options=[""] + brand_options + ["Other"], index=0)
    brand_other = st.text_input("Brand (if Other)", placeholder="Type brand") if brand_choice == "Other" else ""
    brand = brand_other if brand_choice == "Other" else brand_choice
    condition = st.selectbox(
        "Condition",
        options=["", "New", "Open Box", "Used", "Certified - Refurbished"],
        index=0,
    )
    screen_size_in = st.number_input("Screen size (inches)", min_value=10.0, max_value=60.0, value=27.0)
    resolution_width = st.number_input("Resolution width", min_value=640.0, max_value=7680.0, value=2560.0)
    resolution_height = st.number_input("Resolution height", min_value=480.0, max_value=4320.0, value=1440.0)
    refresh_rate_hz = st.number_input("Refresh rate (Hz)", min_value=30.0, max_value=360.0, value=144.0)

    st.subheader("Display details (optional)")
    panel_type = st.selectbox(
        "Panel type",
        options=["", "IPS", "VA", "TN", "OLED"],
        index=0,
    )
    aspect_ratio = st.selectbox(
        "Aspect ratio",
        options=["", "16:9", "21:9", "32:9", "16:10", "4:3"],
        index=0,
    )
    response_time_ms = st.number_input("Response time (ms)", min_value=0.0, max_value=20.0, value=1.0)
    hdr = st.selectbox("HDR", options=["", "Yes", "No"], index=0)
    has_adaptive_sync = st.selectbox("Adaptive sync", options=["", "Yes", "No"], index=0)

    st.subheader("Listing details (optional)")
    shipping_cost = st.number_input("Shipping cost", min_value=0.0, value=0.0)
    color = st.text_input("Color", placeholder="Black")

    submit = st.form_submit_button("Estimate price")


if submit:
    required_artifacts = [
        "artifacts/model.pkl",
        "artifacts/preprocessor.pkl",
        "artifacts/price_stats.pkl",
    ]
    missing = [p for p in required_artifacts if not os.path.exists(p)]
    if missing:
        st.error("Model artifacts are missing. Please run training first.")
        st.write("Missing:", ", ".join(missing))
        st.stop()

    missing_required = []
    if not brand:
        missing_required.append("brand")
    if not condition:
        missing_required.append("condition")
    if not screen_size_in:
        missing_required.append("screen_size_in")
    if not resolution_width:
        missing_required.append("resolution_width")
    if not resolution_height:
        missing_required.append("resolution_height")
    if not refresh_rate_hz:
        missing_required.append("refresh_rate_hz")

    if missing_required:
        st.error(f"Please fill required fields: {', '.join(missing_required)}")
        st.stop()

    payload = {
        "brand": brand or None,
        "condition": condition or None,
        "screen_size_in": screen_size_in or None,
        "resolution_width": resolution_width or None,
        "resolution_height": resolution_height or None,
        "refresh_rate_hz": refresh_rate_hz or None,
        "panel_type": panel_type or None,
        "aspect_ratio": aspect_ratio or None,
        "response_time_ms": response_time_ms or None,
        "hdr": hdr or None,
        "has_adaptive_sync": has_adaptive_sync or None,
        "shipping_cost": shipping_cost,
        "color": color or None,
    }

    try:
        pipeline = PredictPipeline()
        details = pipeline.predict_with_details_from_dict(payload)[0]
        st.success(f"Estimated price: ${details['prediction']:.2f}")
        if "range_low" in details and "range_high" in details:
            st.markdown(
                "<div>"
                "<strong>Suggested range:</strong> "
                f"<span class='value-accent'>${details['range_low']:.2f}</span> – "
                f"<span class='value-accent'>${details['range_high']:.2f}</span>"
                "</div>",
                unsafe_allow_html=True,
            )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
