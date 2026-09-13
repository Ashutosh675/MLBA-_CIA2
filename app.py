import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="🏨",
    layout="wide"
)

# Load model and dataset
model = joblib.load("model.pkl")
df = pd.read_csv("data/hotel_booking.csv")


# Navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Business Problem", "Data Insights", "Prediction"]
)


# =========================================================
# BUSINESS PROBLEM
# =========================================================

if page == "Business Problem":

    st.title("🏨 Hotel Booking Cancellation Predictor")

    st.write(
        "Predict whether a hotel booking is likely to be cancelled "
        "using a machine learning model."
    )

    st.header("Business Problem")

    st.subheader("Problem")

    st.write(
        "Hotel booking cancellations can create uncertainty in room "
        "availability and make it difficult for hotels to plan operations."
    )

    st.subheader("Objective")

    st.write(
        "The objective of this project is to predict whether a hotel "
        "booking is likely to be cancelled using machine learning."
    )

    st.subheader("Dataset")

    st.write("Hotel Booking Demand Dataset")

    st.write("Target variable: is_canceled")


# =========================================================
# DATA INSIGHTS
# =========================================================

elif page == "Data Insights":

    st.title("📊 Data Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Bookings", f"{len(df):,}")

    with col2:
        st.metric("Number of Features", df.shape[1])

    with col3:
        cancellation_rate = df["is_canceled"].mean() * 100
        st.metric("Cancellation Rate", f"{cancellation_rate:.2f}%")

    st.subheader("Booking Cancellation Distribution")

    cancellation_counts = df["is_canceled"].value_counts()
    st.bar_chart(cancellation_counts)


    st.subheader("Cancellation by Hotel Type")

    hotel_cancellation = pd.crosstab(
    df["hotel"],
    df["is_canceled"]
)

    hotel_cancellation.columns = ["Not Cancelled", "Cancelled"]

    st.bar_chart(hotel_cancellation)


    st.subheader("Cancellation Rate by Lead Time")

# Create lead-time groups
    bins = [0, 30, 60, 90, 180, 365, float("inf")]
    labels = [
    "0–30 days",
    "31–60 days",
    "61–90 days",
    "91–180 days",
    "181–365 days",
    "365+ days"
]

    df["lead_time_group"] = pd.cut(
    df["lead_time"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# Calculate cancellation rate
    lead_time_rate = (
    df.groupby("lead_time_group", observed=True)["is_canceled"]
      .mean()
      .mul(100)
)

    st.bar_chart(lead_time_rate)


    st.subheader("Cancellation Rate by Market Segment")

    market_segment_rate = (
    df.groupby("market_segment")["is_canceled"]
      .mean()
      .mul(100)
      .sort_values(ascending=False)
)

    st.bar_chart(market_segment_rate)


# =========================================================
# PREDICTION
# =========================================================

elif page == "Prediction":

    st.title("🔮 Booking Cancellation Prediction")

    st.write(
        "Enter booking details to estimate the likelihood of cancellation."
    )

    st.header("Booking Cancellation Prediction")

    st.write(
    "Enter the booking details below to estimate the likelihood "
    "that the reservation will be cancelled."
)

    hotel = st.selectbox(
    "Hotel Type",
    ["City Hotel", "Resort Hotel"]
)

    lead_time = st.number_input(
    "Lead Time (days)",
    min_value=0,
    max_value=1000,
    value=30
)

    arrival_date_month = st.selectbox(
    "Arrival Month",
    [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]
)

    market_segment = st.selectbox(
    "Market Segment",
    [
        "Online TA",
        "Offline TA/TO",
        "Groups",
        "Direct",
        "Corporate",
        "Complementary",
        "Aviation",
        "Undefined"
    ]
)

    adults = st.number_input(
    "Number of Adults",
    min_value=1,
    max_value=10,
    value=2
)

    stays_in_week_nights = st.number_input(
    "Week Nights",
    min_value=0,
    max_value=30,
    value=2
)

    total_of_special_requests = st.number_input(
    "Special Requests",
    min_value=0,
    max_value=10,
    value=0
)
    st.divider()

    if st.button("🔮 Predict Cancellation", type="primary"):

        # Create one booking record using the values entered above
        input_data = pd.DataFrame([{
            "hotel": hotel,
            "lead_time": lead_time,
            "arrival_date_year": 2017,
            "arrival_date_month": arrival_date_month,
            "arrival_date_week_number": 1,
            "stays_in_weekend_nights": 0,
            "stays_in_week_nights": stays_in_week_nights,
            "adults": adults,
            "children": 0,
            "babies": 0,
            "meal": "BB",
            "country": "PRT",
            "market_segment": market_segment,
            "distribution_channel": "TA/TO",
            "is_repeated_guest": 0,
            "previous_cancellations": 0,
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": "A",
            "deposit_type": "No Deposit",
            "days_in_waiting_list": 0,
            "customer_type": "Transient",
            "adr": 100,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": total_of_special_requests
        }])

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Booking is likely to be CANCELLED")
        else:
            st.success("✅ Booking is likely to be NOT CANCELLED")

        st.metric(
            "Cancellation Probability",
            f"{probability * 100:.2f}%"
        )

        st.subheader("Business Recommendation")

        if prediction == 1:
            st.warning(
                "Consider taking preventive action, such as confirming the "
                "reservation with the guest or applying an appropriate "
                "cancellation policy."
            )
        else:
            st.info(
                "The booking appears relatively stable. Normal reservation "
                "management can be followed."
            )