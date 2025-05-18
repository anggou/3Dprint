import streamlit as st
import pandas as pd
import numpy as np
from pymcdm.methods import TOPSIS
from sklearn.neural_network import MLPRegressor
import openpyxl
import io

# ------------------------
# 기준 설정
# ------------------------
criteria = ["Material Properties", "Required Precision", "Operational Impact", "Procurement Cost", "Accessibility", "Specific Volume"]
types = np.array([1, 1, 1, -1, -1, -1])  # 1: 클수록 좋음, -1: 작을수록 좋음

# ------------------------
# 초기 세션 상태 정의
# ------------------------
def init_session():
    if "spare_parts" not in st.session_state:
        st.session_state.spare_parts = {}
    if "price_series" not in st.session_state:
        st.session_state.price_series = []

# ------------------------
# MLP 기반 TOPSIS 가중치 예측
# ------------------------
def predict_weights_from_price_series(series):
    if len(series) < 6:
        return np.array([0.25, 0.2, 0.25, 0.1, 0.1, 0.1])  # 데이터 부족 시 기본값

    # 학습용 예시 데이터셋
    X = np.array([
        [5.2, 5.4, 5.5, 5.8, 6.1, 6.3],
        [3.1, 3.0, 3.2, 3.3, 3.2, 3.3],
        [6.2, 6.4, 6.5, 6.6, 6.8, 7.0],
        [8.0, 8.2, 8.1, 8.3, 8.4, 8.6],
        [4.0, 4.1, 4.2, 4.3, 4.5, 4.7],
        [7.0, 7.1, 7.2, 7.5, 7.8, 8.0],
    ])
    y = np.array([
        [0.15, 0.2, 0.25, 0.2, 0.1, 0.1],
        [0.1, 0.15, 0.3, 0.25, 0.1, 0.1],
        [0.25, 0.25, 0.2, 0.15, 0.1, 0.05],
        [0.3, 0.25, 0.15, 0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
        [0.28, 0.2, 0.2, 0.12, 0.1, 0.1],
    ])

    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    model.fit(X, y)

    test_input = np.array(series[-6:]).reshape(1, -1)
    predicted = model.predict(test_input)[0]
    return predicted / np.sum(predicted)

# ------------------------
# TOPSIS 평가 함수
# ------------------------
def evaluate_topsis(user_part, weights):
    alternatives = np.array([
        [9, 9, 8, 9, 9, 5],
        [8, 6, 10, 8, 10, 6],
        [7, 3, 7, 4, 3, 8]
    ])
    labels = ["Manufacturing", "Emergency Manufacturing", "Shipping Request"]

    user_part_array = np.array(user_part).reshape(1, -1)
    decision_matrix = np.vstack([user_part_array, alternatives])

    topsis = TOPSIS()
    scores = topsis(decision_matrix, weights, types)
    user_score = scores[0]
    alt_scores = scores[1:]
    diffs = np.abs(alt_scores - user_score)
    closest_index = np.argmin(diffs)

    return labels[closest_index], alt_scores[closest_index]

# ------------------------
# Streamlit UI
# ------------------------
init_session()
st.title("🛠️ Onboard Manufacturing Decision Platform with ML-based TOPSIS")

menu = st.sidebar.selectbox("기능 선택", ["Add Spare", "Production evaluation", "See Spare List", "Import/Export Data", "Input Price Series"])

# 부품 추가
if menu == "Add Spare":
    with st.form("부품 입력"):
        name = st.text_input("Spare Part Name:")
        values = [st.slider(c, 0, 10, 5) for c in criteria]
        submitted = st.form_submit_button("Add")
        if submitted:
            if name:
                st.session_state.spare_parts[name] = values
                st.success(f"{name} Add Complete!")
            else:
                st.warning("Input Spare Part Name.")

# 평가 수행
elif menu == "Production evaluation":
    part_list = list(st.session_state.spare_parts.keys())
    if part_list:
        selected = st.selectbox("평가할 부품 선택", part_list)
        if selected:
            user_part = st.session_state.spare_parts[selected]
            st.write(f"선택된 부품: {selected}")
            st.write("입력된 값:", dict(zip(criteria, user_part)))

            ml_weights = predict_weights_from_price_series(st.session_state.price_series)
            st.info(f"ML 기반 가중치: {np.round(ml_weights, 3)}")

            selected_alternative, similarity_score = evaluate_topsis(user_part, ml_weights)
            st.success(f"이 부품은 **'{selected_alternative}'** 대안에 가장 유사합니다. (유사도 점수: {similarity_score:.4f})")
    else:
        st.warning("먼저 부품을 하나 이상 등록하세요.")

# 부품 목록 보기
elif menu == "See Spare List":
    if st.session_state.spare_parts:
        st.subheader("🔧 현재 등록된 Spare List")
        for part_name, values in st.session_state.spare_parts.items():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**{part_name}**: {dict(zip(criteria, values))}")
            with col2:
                if st.button("삭제", key=f"del_{part_name}"):
                    del st.session_state.spare_parts[part_name]
                    st.success(f"{part_name} 삭제 완료.")
                    st.experimental_rerun()
    else:
        st.info("아직 입력된 부품이 없습니다.")

# Import/Export
elif menu == "Import/Export Data":
    st.subheader("📥 Import Data from Excel")
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    if uploaded_file:
        try:
            imported_data = pd.read_excel(uploaded_file, index_col=0)
            st.session_state.spare_parts = imported_data.to_dict(orient='index')
            st.success("✅ Data imported successfully!")
            st.dataframe(imported_data)
        except Exception as e:
            st.error(f"❌ Error while uploading file: {e}")

    st.subheader("📤 Export Data to Excel")
    if st.session_state.spare_parts:
        export_df = pd.DataFrame.from_dict(st.session_state.spare_parts, orient='index', columns=criteria)
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=True)
        buffer.seek(0)
        st.download_button(
            label="Download Excel File",
            data=buffer,
            file_name="spare_parts.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("No data available to export.")

# 조달가격 시계열 입력
elif menu == "Input Price Series":
    st.subheader("📈 시계열 조달가격 입력")
    new_price = st.number_input("이번 분기 조달가격 입력", min_value=0.0, step=0.1)
    if st.button("Add Price"):
        st.session_state.price_series.append(new_price)
        st.success(f"{new_price} 추가 완료")
    if st.session_state.price_series:
        st.line_chart(st.session_state.price_series)
