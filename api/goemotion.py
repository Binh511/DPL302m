# goemotions.py

import io
import time
import gc
import psutil
import pandas as pd
import uvicorn

from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB


class ModelReport(BaseModel):
    model_name: str
    accuracy: float
    latency_seconds: float
    cost_ram_mb: float

class EvaluationResponse(BaseModel):
    reports: List[ModelReport]
    baseline_ram_mb: float



app = FastAPI()

def get_memory_usage_mb() -> float:
    process = psutil.Process()
    # process.memory_info().rss trả về số bytes, chia cho (1024*1024) để đổi sang MB
    return process.memory_info().rss / (1024 * 1024)

def prepare_data(df: pd.DataFrame) -> tuple:
    """
    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu thô.
    Returns:
        tuple: Chứa (X_train, X_test, y_train, y_test).
    """
    all_columns = df.columns.tolist()
    try:
        start_index = all_columns.index('admiration')
        emotion_columns = all_columns[start_index:]
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Các cột cảm xúc bắt buộc không được tìm thấy trong file CSV."
        )

    if 'text' not in df.columns:
        raise HTTPException(status_code=400, detail="Cột 'text' không được tìm thấy trong file CSV.")
    
    X = df['text']
    y = df[emotion_columns]

    # Chia dữ liệu thành 80% train và 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def evaluate_single_model(
    model: Any,
    model_name: str,
    X_train: pd.Series,
    y_train: pd.DataFrame,
    X_test: pd.Series,
    y_test: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Args:
        model: Một instance của model sklearn.
        model_name (str): Tên của model để hiển thị.
        X_train, y_train, X_test, y_test: Dữ liệu đã được chia.
    Returns:
        dict: Chứa pipeline đã được huấn luyện, accuracy và latency.
    """

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', OneVsRestClassifier(model, n_jobs=-1)) 
    ])

    # Đo lường latency
    start_time = time.perf_counter()
    
    # Huấn luyện model
    pipeline.fit(X_train, y_train)
    
    # Dự đoán trên tập test
    predictions = pipeline.predict(X_test)
    
    end_time = time.perf_counter()
    latency = end_time - start_time
    
    # Tính accuracy 
    accuracy = accuracy_score(y_test, predictions)
    
    return {
        "pipeline": pipeline,
        "accuracy": accuracy,
        "latency": latency,
    }


@app.post("/evaluate_models", response_model=EvaluationResponse)
async def evaluate_models(file: UploadFile = File(...)):
    # Kiểm tra định dạng file
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Định dạng file không hợp lệ. Vui lòng upload file .csv.")
    
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi khi đọc file CSV: {e}")

    # Chuẩn bị dữ liệu
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Định nghĩa các model cần đánh giá
    models_to_evaluate = [
        ("Logistic Regression", LogisticRegression(solver='liblinear', random_state=42)),
        ("Linear SVM", LinearSVC(random_state=42, dual=False)), # dual=False is recommended when n_samples > n_features
        ("Naive Bayes", MultinomialNB()),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ]
    
    reports = []


    baseline_ram_mb = get_memory_usage_mb()
    for name, model_instance in models_to_evaluate:
        loop_baseline_ram = get_memory_usage_mb()

        # Huấn luyện và đánh giá
        result = evaluate_single_model(model_instance, name, X_train, y_train, X_test, y_test)
        

        ram_after_fit = get_memory_usage_mb()
        cost_ram = ram_after_fit - loop_baseline_ram
        
        # Tạo báo cáo
        report = ModelReport(
            model_name=name,
            accuracy=result["accuracy"],
            latency_seconds=result["latency"],
            cost_ram_mb=cost_ram
        )
        reports.append(report)
        

        # Xóa cache
        del result['pipeline'] 
        gc.collect()
        
    return EvaluationResponse(reports=reports, baseline_ram_mb=baseline_ram_mb)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)