import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predictive Return Management System", page_icon="📦", layout="wide")

st.title("📦 Predictive Return Management System")
st.markdown("This app predicts the likelihood of a product return based on order and delivery details.")

@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    return model

model = load_model()

st.sidebar.header("Upload Order Data")
uploaded_file = st.sidebar.file_uploader("Upload your order CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(data.head())

    required_features = ["Order_Value_INR", "Priority", "Product_Category", 
                         "Customer_Segment", "Promised_Delivery_Days", 
                         "Actual_Delivery_Days", "Distance_KM"]
    missing = [col for col in required_features if col not in data.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    else:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in ["Priority", "Product_Category", "Customer_Segment"]:
            data[col] = le.fit_transform(data[col].astype(str))

        predictions = model.predict(data[required_features])
        data["Return_Risk"] = ["Likely Return" if p == 1 else "No Return" for p in predictions]

        st.success("✅ Predictions complete!")
        st.write("### Prediction Results")
        st.dataframe(data[["Order_Value_INR", "Priority", "Product_Category", "Return_Risk"]])

        st.write("### Return Risk Distribution")
        fig, ax = plt.subplots()
        data["Return_Risk"].value_counts().plot(kind="bar", color=["skyblue", "salmon"], ax=ax)
        plt.xticks(rotation=0)
        st.pyplot(fig)

        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results CSV", data=csv, file_name="predicted_returns.csv", mime="text/csv")

else:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
