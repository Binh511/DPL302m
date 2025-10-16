import re

import pandas as pd

# 1️⃣ Đọc dữ liệu
df = pd.read_csv("goemotions_merged.csv")

print("Kích thước dữ liệu ban đầu:", df.shape)
print("Các cột trong dataset:", df.columns.tolist()[:10], "...")

# 2️⃣ Loại bỏ dòng trống hoặc lỗi
df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str)
df = df[df["text"].str.strip() != ""]


# 3️⃣ Hàm làm sạch text
def clean_text(text):
    # lowercase
    text = text.lower()
    # xóa URL
    text = re.sub(r"http\S+|www.\S+", "", text)
    # xóa mentions và hashtag
    text = re.sub(r"@\w+|#\w+", "", text)
    # xóa ký tự không phải chữ/số, emoji, punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # rút gọn nhiều khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Áp dụng cleaning
df["clean_text"] = df["text"].apply(clean_text)

# 4️⃣ Loại bỏ dòng trống sau cleaning
df = df[df["clean_text"] != ""]

# 5️⃣ Xóa trùng lặp (nếu có)
df = df.drop_duplicates(subset="clean_text")

print("Sau làm sạch:", df.shape)

# 6️⃣ Kiểm tra vài dòng
print("\n🔹 Mẫu dữ liệu sau cleaning:")
print(df[["text", "clean_text"]].head(5))

# 7️⃣ (Tùy chọn) Phân tích nhãn
label_cols = [col for col in df.columns if col not in ["text", "clean_text"]]
if label_cols:
    print("\nSố lượng nhãn:", len(label_cols))
    # Chuyển các cột label về kiểu số nguyên để tránh lỗi sort_values
    df[label_cols] = (
        df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    )
    label_counts = df[label_cols].sum().sort_values(ascending=False)
    print("\nPhân phối nhãn (top 10):")
    print(label_counts.head(10))
    print("\nSố nhãn trung bình mỗi mẫu:", df[label_cols].sum(axis=1).mean().round(2))

# 8️⃣ Lưu dữ liệu sạch
df.to_csv("cleaned_goemotions.csv", index=False)
print("\n✅ Đã lưu file 'cleaned_goemotions.csv'")
