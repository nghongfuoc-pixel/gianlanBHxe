import streamlit as st
import pandas as pd
import shutil
from huggingface_hub import hf_hub_download
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title="Dự đoán Gian lận Bảo hiểm Ô tô", page_icon="🚗")

st.title("🚗 Công cụ Dự đoán Gian lận Bảo hiểm Ô tô")
st.write(
    "Upload file CSV chứa thông tin claim (đúng định dạng dữ liệu huấn luyện, "
    "không cần cột FRAUD) để nhận ngay xác suất gian lận cho từng dòng."
)


@st.cache_resource
def tai_model():
    # Tải model từ Hugging Face Hub (repo của anh/chị) về máy chủ Streamlit
    # Thay "hongphuoc/glbhx" nếu tên repo model khác
    downloaded_path = hf_hub_download(repo_id="hongphuoc/glbhx", filename="fraud_model.pkl")
    shutil.copy(downloaded_path, "fraud_model.pkl")
    # PyCaret cần tên không có đuôi .pkl (tự thêm vào)
    return load_model("fraud_model")


model = tai_model()

uploaded_file = st.file_uploader("Chọn file CSV cần dự đoán", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_predict = df.drop(columns=["CLAIM_DESCRIPTION"], errors="ignore")

    with st.spinner("Đang dự đoán..."):
        result = predict_model(model, data=df_predict)

    output_cols = ["ID", "prediction_label", "prediction_score"]
    result_view = result[[c for c in output_cols if c in result.columns]]

    st.subheader("Kết quả dự đoán")
    st.dataframe(result_view, use_container_width=True)

    csv_bytes = result_view.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="📥 Tải file kết quả CSV",
        data=csv_bytes,
        file_name="ket_qua_du_doan.csv",
        mime="text/csv",
    )
