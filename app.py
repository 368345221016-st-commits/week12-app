from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")


# ---------- โหลดโมเดล (โหลดครั้งเดียว แล้วจำไว้) ----------
@st.cache_resource
def load_model():
    # หาไฟล์ titanic_tree*.joblib ในโฟลเดอร์เดียวกับ app.py
    # (รองรับทั้งชื่อ titanic_tree.joblib และ "titanic_tree (1).joblib")
    folder = Path(__file__).parent
    files = sorted(folder.glob("titanic_tree*.joblib"))
    if not files:
        return None, None
    return joblib.load(files[0]), files[0].name


model, model_file = load_model()

st.title("🚢 Titanic Survival Predictor")

if model is None:
    st.error(
        "ไม่พบไฟล์โมเดล `titanic_tree*.joblib` ในโฟลเดอร์เดียวกับ app.py\n\n"
        "กรุณาคัดลอกไฟล์โมเดลมาวางไว้ในโฟลเดอร์เดียวกับ app.py แล้วรีเฟรชหน้านี้"
    )
    st.stop()

st.caption(f"โมเดลที่ใช้: {model_file} (Decision Tree)")

# ---------- ฟอร์มรับข้อมูลผู้โดยสาร ----------
st.subheader("ข้อมูลผู้โดยสาร")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "ชั้นตั๋ว (Pclass)",
        options=[1, 2, 3],
        index=2,
        format_func=lambda x: {1: "1 - ชั้นหนึ่ง", 2: "2 - ชั้นสอง", 3: "3 - ชั้นสาม"}[x],
    )
    sex = st.radio("เพศ", options=["ชาย", "หญิง"], horizontal=True)
    age = st.number_input("อายุ (ปี)", min_value=0.0, max_value=100.0, value=30.0, step=1.0)

with col2:
    fare = st.number_input(
        "ค่าโดยสาร (Fare)", min_value=0.0, max_value=600.0, value=32.0, step=1.0
    )
    family_size = st.number_input(
        "ขนาดครอบครัว (FamilySize)",
        min_value=1,
        max_value=15,
        value=1,
        step=1,
        help="จำนวนสมาชิกครอบครัวที่เดินทางด้วยกัน รวมตัวผู้โดยสารเอง (เดินทางคนเดียว = 1)",
    )

# ---------- ทำนาย ----------
if st.button("ทำนายผล", type="primary"):
    # ชื่อคอลัมน์และลำดับต้องตรงกับตอนเทรนโมเดลเสมอ
    X = pd.DataFrame(
        [
            {
                "Pclass": pclass,
                "Sex_female": 1 if sex == "หญิง" else 0,
                "Age": age,
                "Fare": fare,
                "FamilySize": family_size,
            }
        ]
    )
    X = X[list(model.feature_names_in_)]

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    p_survive = float(proba[list(model.classes_).index(1)])

    st.divider()
    if pred == 1:
        st.success("ผลทำนาย: **รอดชีวิต** ✅")
    else:
        st.error("ผลทำนาย: **ไม่รอดชีวิต** ❌")

    st.metric("ความน่าจะเป็นที่จะรอดชีวิต", f"{p_survive:.1%}")
    st.progress(p_survive)

    with st.expander("ดูข้อมูลที่ส่งเข้าโมเดล"):
        st.dataframe(X)
