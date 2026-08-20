import gradio as gr
import pandas as pd
import shutil
from huggingface_hub import hf_hub_download
from pycaret.classification import load_model, predict_model

# Tải model từ Hugging Face Hub (repo của anh/chị) về máy chủ Space
# Thay "hongphuoc/glbhx" nếu tên repo model khác
downloaded_path = hf_hub_download(repo_id="hongphuoc/glbhx", filename="fraud_model.pkl")
shutil.copy(downloaded_path, "fraud_model.pkl")

# PyCaret cần tên không có đuôi .pkl (tự thêm vào)
model = load_model("fraud_model")


def du_doan(file):
    df = pd.read_csv(file.name)
    df_predict = df.drop(columns=['CLAIM_DESCRIPTION'], errors='ignore')

    result = predict_model(model, data=df_predict)

    output_cols = ['ID', 'prediction_label', 'prediction_score']
    result_view = result[[c for c in output_cols if c in result.columns]]

    output_path = "ket_qua_du_doan.csv"
    result_view.to_csv(output_path, index=False)

    return result_view, output_path


demo = gr.Interface(
    fn=du_doan,
    inputs=gr.File(label="Upload file CSV dữ liệu claim cần dự đoán"),
    outputs=[
        gr.Dataframe(label="Kết quả dự đoán"),
        gr.File(label="Tải file kết quả CSV")
    ],
    title="🚗 Công cụ Dự đoán Gian lận Bảo hiểm Ô tô",
    description=(
        "Upload file CSV chứa thông tin claim (đúng định dạng dữ liệu huấn luyện, "
        "không cần cột FRAUD) để nhận ngay xác suất gian lận cho từng dòng."
    ),
)

if __name__ == "__main__":
    demo.launch()