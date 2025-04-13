import streamlit as st
import pandas as pd
import numpy as np
from pymcdm.methods import TOPSIS
import openpyxl
import io  # 메모리 기반 파일 처리를 위한 모듈

# 기준과 설정
criteria = ["Material Properties", "Required Precision", "Operational Impact", "Procurement Cost", "Accessibility",
            "Specific Volume"]
weights = np.array([0.25, 0.2, 0.25, 0.1, 0.1, 0.1])
cost_criteria = np.array([0, 0, 0, 1, 0, 1])


# 세션 상태에 부품 리스트 저장
def init_session():
    if "spare_parts" not in st.session_state:
        st.session_state.spare_parts = {}


# 가장 유사한 대안 1개만 선택하는 TOPSIS 평가 함수
def evaluate_topsis(user_part):
    alternatives = np.array([
        [8, 7, 6, 6, 7, 5],  # Manufacturing
        [7, 8, 8, 7, 8, 6],  # Emergency Manufacturing
        [7, 7, 9, 5, 9, 8]  # Shipping
    ])
    labels = ["Manufacturing", "Emergency Manufacturing", "Shipping Request"]

    user_part_array = np.array(user_part).reshape(1, -1)
    decision_matrix = np.vstack([user_part_array, alternatives])

    topsis = TOPSIS()
    scores = topsis(decision_matrix, weights, cost_criteria)

    # 사용자 입력은 0번, 대안은 1~3번 인덱스
    user_score = scores[0]
    alt_scores = scores[1:]
    diffs = np.abs(alt_scores - user_score)
    closest_index = np.argmin(diffs)

    return labels[closest_index], alt_scores[closest_index]


# UI 시작
init_session()
st.title("Onboard Manufacturing Decision Platform")

menu = st.sidebar.selectbox("기능 선택", ["Add Spare", "Production evaluation", "See Spare List", "Import/Export Data"])

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

elif menu == "Production evaluation":
    part_list = list(st.session_state.spare_parts.keys())
    if part_list:
        selected = st.selectbox("평가할 부품 선택", part_list)
        if selected:
            user_part = st.session_state.spare_parts[selected]
            st.write(f"선택된 부품: {selected}")
            st.write("입력된 값:", dict(zip(criteria, user_part)))

            selected_alternative, similarity_score = evaluate_topsis(user_part)
            st.success(f"이 부품은 **'{selected_alternative}'** 대안에 가장 유사합니다. (유사도 점수: {similarity_score:.4f})")
    else:
        st.warning("먼저 부품을 하나 이상 등록하세요.")

elif menu == "See Spare List":
    if st.session_state.spare_parts:
        st.subheader("현재 등록된 Spare List")
        for part_name, values in st.session_state.spare_parts.items():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"🔧 **{part_name}**: {dict(zip(criteria, values))}")
            with col2:
                if st.button("삭제", key=f"del_{part_name}"):
                    del st.session_state.spare_parts[part_name]
                    st.success(f"{part_name}이(가) 삭제되었습니다.")
                    st.experimental_rerun()  # 삭제 후 새로고침
    else:
        st.info("아직 입력된 부품이 없습니다.")


elif menu == "Import/Export Data":
    st.subheader("Import Data from Excel")
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    if uploaded_file:
        try:
            imported_data = pd.read_excel(uploaded_file, index_col=0)
            st.session_state.spare_parts = imported_data.to_dict(orient='index')
            st.success("Data imported successfully!")
            st.dataframe(imported_data)
        except Exception as e:
            st.error(f"Error occurred while uploading the file: {e}")

    st.subheader("Export Data to Excel")
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
