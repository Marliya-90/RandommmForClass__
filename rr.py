import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Sales Prediction App", layout="centered")

st.title("📊 Sales Prediction - Regression App")
st.write("Upload your **sales_data.csv** file")

# Upload CSV
file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    # Only numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("❌ Dataset must have at least 2 numeric columns")
    else:
        st.subheader("⚙️ Select Columns")

        target = st.selectbox("Select Target Column (Y)", numeric_cols)

        features = st.multiselect(
            "Select Feature Columns (X)",
            [col for col in numeric_cols if col != target]
        )

        if len(features) == 0:
            st.warning("⚠️ Please select at least one feature column")
        else:
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)

            st.success("✅ Model Trained Successfully")
            st.write(f"📈 R² Score: **{score:.2f}**")

            st.subheader("🧮 Make a Prediction")

            input_data = []
            for col in features:
                val = st.number_input(f"Enter {col}", value=0.0)
                input_data.append(val)

            if st.button("Predict Sales"):
                prediction = model.predict([input_data])
                st.success(f"💰 Predicted Sales Value: {prediction[0]:.2f}")
