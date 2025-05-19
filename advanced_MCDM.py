import streamlit as st
import pandas as pd
import numpy as np
from pymcdm.methods import TOPSIS
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import openpyxl
import io

# 기준 설정
criteria = ["Material Properties", "Required Precision", "Operational Impact",
            "Procurement Cost", "Accessibility", "Specific Volume"]
weights = np.array([0.25, 0.2, 0.25, 0.1, 0.1, 0.1])
cost_criteria = np.array([0, 0, 0, 1, 0, 1])
labels = ["Manufacturing", "Emergency Manufacturing", "Shipping Request"]

# 세션 초기화
if "spare_parts" not in st.session_state:
    st.session_state.spare_parts = {}

# TOPSIS 평가 함수
def evaluate_topsis(user_part):
    alternatives = np.array([
        [8, 7, 6, 6, 7, 5],  # Manufacturing
        [7, 8, 8, 7, 8, 6],  # Emergency Manufacturing
        [7, 7, 9, 5, 9, 8]   # Shipping
    ])

    user_array = np.array(user_part).reshape(1, -1)
    decision_matrix = np.vstack([user_array, alternatives])
    topsis = TOPSIS()
    scores = topsis(decision_matrix, weights, cost_criteria)
    user_score = scores[0]
    alt_scores = scores[1:]
    diffs = np.abs(alt_scores - user_score)
    best_idx = np.argmin(diffs)
    return labels[best_idx], alt_scores[best_idx]

# SHAP 시각화 함수
def show_shap_explanation(X_input, model):
    explainer = shap.Explainer(model, X_input)
    shap_values = explainer(X_input)
    st.subheader("XAI (SHAP) 중요도 시각화")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

# Streamlit 인터페이스
st.title("3D Printing Decision Framework with XAI (SHAP)")
menu = st.sidebar.selectbox("메뉴 선택", ["Spare 등록", "제조 평가", "리스트 보기", "데이터 가져오기/내보내기"])

if menu == "Spare 등록":
    with st.form("부품 입력"):
        name = st.text_input("Spare Part Name:")
        values = [st.slider(c, 0, 10, 5) for c in criteria]
        if st.form_submit_button("추가"):
            if name:
                st.session_state.spare_parts[name] = values
                st.success(f"{name} 등록 완료")
            else:
                st.warning("부품 이름을 입력하세요.")

elif menu == "제조 평가":
    if st.session_state.spare_parts:
        selected = st.selectbox("평가할 부품 선택", list(st.session_state.spare_parts.keys()))
        user_part = st.session_state.spare_parts[selected]
        st.write("입력된 특성:", dict(zip(criteria, user_part)))

        result, score = evaluate_topsis(user_part)
        st.markdown(f"### 결과: **{result}** (유사도 점수: {score:.4f})")

        # SHAP 시각화를 위한 데이터 및 모델 준비
        alt_data = [
            [8, 7, 6, 6, 7, 5],
            [7, 8, 8, 7, 8, 6],
            [7, 7, 9, 5, 9, 8]
        ]
        alt_labels = [0, 1, 2]  # 대안 인덱스를 label로 가정
        rf_model = RandomForestClassifier().fit(alt_data, alt_labels)
        show_shap_explanation(pd.DataFrame([user_part], columns=criteria), rf_model)
    else:
        st.warning("먼저 부품을 등록하세요.")

elif menu == "리스트 보기":
    if st.session_state.spare_parts:
        for part, vals in st.session_state.spare_parts.items():
            st.write(f"🔧 **{part}**: {dict(zip(criteria, vals))}")
    else:
        st.info("등록된 부품이 없습니다.")

elif menu == "데이터 가져오기/내보내기":
    st.subheader("📥 Import from Excel")
    uploaded = st.file_uploader("Upload Excel", type=["xlsx"])
    if uploaded:
        try:
            df = pd.read_excel(uploaded, index_col=0)
            st.session_state.spare_parts = df.to_dict(orient='index')
            st.success("불러오기 완료")
            st.dataframe(df)
        except Exception as e:
            st.error(f"오류 발생: {e}")

    st.subheader("📤 Export to Excel")
    if st.session_state.spare_parts:
        df_out = pd.DataFrame.from_dict(st.session_state.spare_parts, orient='index', columns=criteria)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_out.to_excel(writer)
        buf.seek(0)
        st.download_button("Download Excel", data=buf, file_name="spare_parts.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.info("내보낼 데이터가 없습니다.")
