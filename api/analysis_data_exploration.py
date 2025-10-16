"""
Data Exploration Script
1. Data Inspection (Missing Values, Data Types)
2. Label distribution
3. Tokenize, N-top words

"""

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
warnings.filterwarnings("ignore")


def load_raw_data():
    """Load and merge the three CSV files"""
    path1 = "D:\DPL\goemotions_1.csv"
    path2 = "goemotions_2.csv"
    path3 = "goemotions_3.csv"

    # Read all CSV files
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df3 = pd.read_csv(path3)

    # Concatenate them
    data = pd.concat([df1, df2, df3], ignore_index=True)
    print(f"✅ Loaded and merged data: {data.shape}")
    return data


def eda_before_cleaning(data):
    """Perform EDA on raw data to understand what needs cleaning"""
    print("\n" + "=" * 50)
    print("📊 EDA BEFORE DATA CLEANING")
    print("=" * 50)

    # Basic info
    print(f"Dataset shape: {data.shape}")
    print(f"Column names: {list(data.columns)}")

    # Data types
    print("\n📋 Data Types:")
    print(data.dtypes)

    # Missing values
    print("\n❌ Missing Values:")
    missing = data.isnull().sum()
    missing_pct = (missing / len(data)) * 100
    missing_info = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    print(missing_info[missing_info["Missing Count"] > 0])

    # Basic statistics for numerical columns
    print("\n📈 Numerical Columns Statistics:")
    print(data.describe())

    # Text column analysis
    if "text" in data.columns:
        print("\n📝 Text Column Analysis:")
        print(f"Number of texts: {len(data['text'])}")
        print(f"Unique texts: {data['text'].nunique()}")
        print(f"Average text length: {data['text'].str.len().mean():.2f}")
        print(f"Max text length: {data['text'].str.len().max()}")
        print(f"Min text length: {data['text'].str.len().min()}")

    # Check for emotion columns (assuming they are binary)
    emotion_cols = [
        col
        for col in data.columns
        if col
        not in [
            "text",
            "id",
            "author",
            "subreddit",
            "link_id",
            "parent_id",
            "created_utc",
            "rater_id",
            "example_very_unclear",
        ]
    ]
    if emotion_cols:
        print(
            f"\n😊 Emotion Columns ({len(emotion_cols)}): {emotion_cols[:10]}..."
        )  # Show first 10

        # Emotion distribution
        emotion_counts = data[emotion_cols].sum().sort_values(ascending=False)
        print("\n🎭 Emotion Distribution (Top 10):")
        print(emotion_counts.head(10))

        # Multi-label statistics
        labels_per_sample = data[emotion_cols].sum(axis=1)
        print("\n🏷️  Labels per sample statistics:")
        print(f"Average labels per sample: {labels_per_sample.mean():.2f}")
        print(f"Max labels per sample: {labels_per_sample.max()}")
        print(f"Samples with no labels: {(labels_per_sample == 0).sum()}")
        print(f"Samples with 1 label: {(labels_per_sample == 1).sum()}")
        print(f"Samples with >1 label: {(labels_per_sample > 1).sum()}")

    return emotion_cols


def clean_data(data):
    """Clean the dataset by removing unnecessary columns and unclear examples"""
    print("\n" + "=" * 50)
    print("🧹 DATA CLEANING")
    print("=" * 50)

    # Store original shape
    original_shape = data.shape

    # Drop unnecessary columns
    dropped_cols = [
        "id",
        "author",
        "subreddit",
        "link_id",
        "parent_id",
        "created_utc",
        "rater_id",
    ]
    existing_dropped_cols = [col for col in dropped_cols if col in data.columns]

    if existing_dropped_cols:
        data = data.drop(columns=existing_dropped_cols)
        print(f"✂️  Dropped columns: {existing_dropped_cols}")

    # Remove unclear examples
    if "example_very_unclear" in data.columns:
        unclear_count = (data["example_very_unclear"]).sum()
        data = data[data["example_very_unclear"]]
        data = data.drop(columns=["example_very_unclear"])
        print(f"🗑️  Removed {unclear_count} unclear examples")

    # Remove rows with empty text
    if "text" in data.columns:
        empty_text = data["text"].isna().sum()
        data = data.dropna(subset=["text"])
        if empty_text > 0:
            print(f"🗑️  Removed {empty_text} rows with empty text")

    print(f"📊 Shape changed from {original_shape} to {data.shape}")
    return data


def preprocess_text(data, text_column="text"):
    """Preprocess text data"""
    print("\n" + "=" * 50)
    print("🔤 TEXT PREPROCESSING")
    print("=" * 50)

    def clean_text(text):
        if not isinstance(text, str):
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep spaces
        text = re.sub(r"[^a-z\s]", "", text)
        # Remove extra whitespace
        text = " ".join(text.split())
        return text

    # Store original lengths for comparison
    original_lengths = data[text_column].str.len()

    # Apply preprocessing
    data[text_column] = data[text_column].apply(clean_text)

    # Remove empty texts after preprocessing
    empty_after_clean = (data[text_column].str.len() == 0).sum()
    if empty_after_clean > 0:
        data = data[data[text_column].str.len() > 0]
        print(f"🗑️  Removed {empty_after_clean} empty texts after preprocessing")

    # Show preprocessing results
    new_lengths = data[text_column].str.len()
    print(f"📏 Average length before: {original_lengths.mean():.2f}")
    print(f"📏 Average length after: {new_lengths.mean():.2f}")

    return data


def eda_after_cleaning(data, emotion_cols):
    """Perform EDA on cleaned data"""
    print("\n" + "=" * 50)
    print("📊 EDA AFTER DATA CLEANING")
    print("=" * 50)

    # Basic info
    print(f"Final dataset shape: {data.shape}")

    # Missing values check
    missing = data.isnull().sum().sum()
    if missing == 0:
        print("✅ No missing values in cleaned data")
    else:
        print(f"⚠️  Still {missing} missing values")

    # Text statistics after cleaning
    if "text" in data.columns:
        print("\n📝 Cleaned Text Statistics:")
        text_lengths = data["text"].str.len()
        word_counts = data["text"].str.split().str.len()

        print(f"Average text length: {text_lengths.mean():.2f} characters")
        print(f"Average word count: {word_counts.mean():.2f} words")
        print(f"Text length range: {text_lengths.min()} - {text_lengths.max()}")
        print(f"Word count range: {word_counts.min()} - {word_counts.max()}")

    # Final emotion distribution
    if emotion_cols:
        print("\n🎭 Final Emotion Distribution (Top 10):")
        final_emotion_counts = data[emotion_cols].sum().sort_values(ascending=False)
        print(final_emotion_counts.head(10))

        # Multi-label statistics after cleaning
        labels_per_sample = data[emotion_cols].sum(axis=1)
        print("\n🏷️  Final Multi-label Statistics:")
        print(f"Average labels per sample: {labels_per_sample.mean():.2f}")
        print(f"Samples with no labels: {(labels_per_sample == 0).sum()}")
        print(f"Samples with multiple labels: {(labels_per_sample > 1).sum()}")

    # Show some sample texts
    print("\n📖 Sample Cleaned Texts:")
    for i, text in enumerate(data["text"].head(3)):
        print(f"{i+1}. {text}")


def save_cleaned_data(data, filename="Dataset/goemotions_cleaned.csv"):
    """Save the cleaned dataset"""
    data.to_csv(filename, index=False)
    print(f"\n💾 Cleaned data saved to {filename}")


def data_inspection(data):
    dtype_counts = data.dtypes.value_counts().reset_index()
    dtype_counts.columns = ["Data Type", "Number of Columns"]
    print("Data Types Table:\n", dtype_counts.to_string(index=False))
    missing = data.isnull().sum()
    print("\nMissing Values (nonzero only):\n", missing[missing > 0])


def text_inspection(data, text_column="text", top_n=20):
    vectorizer = CountVectorizer(stop_words="english", max_features=top_n)
    X = vectorizer.fit_transform(data[text_column].dropna())
    word_counts = (
        pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        .sum()
        .sort_values(ascending=False)
    )
    print(f"\nTop {top_n} words:\n", word_counts)


def check_label_distribution(data):
    label_columns = data.columns.difference(
        ["text", "text_length", "word_count", "label_count"]
    )
    label_sums = data[label_columns].sum().sort_values(ascending=False)
    print("\nLabel Distribution:\n", label_sums)
    # Plotting
    plt.figure(figsize=(12, 6))
    label_sums.plot(kind="bar")
    plt.title("Label Distribution")
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def text_length_analysis(data, text_column="text"):
    """Analyze text length and word count distributions"""
    data["text_length"] = data[text_column].str.len()
    data["word_count"] = data[text_column].str.split().str.len()

    print("\nText Length Statistics:")
    print(data[["text_length", "word_count"]].describe())

    # Plot distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    data["text_length"].hist(bins=50, ax=ax1, alpha=0.7, color="skyblue")
    ax1.set_title("Character Length Distribution")
    ax1.set_xlabel("Character Count")
    ax1.set_ylabel("Frequency")
    ax1.axvline(
        data["text_length"].mean(),
        color="red",
        linestyle="--",
        label=f'Mean: {data["text_length"].mean():.0f}',
    )
    ax1.legend()

    data["word_count"].hist(bins=50, ax=ax2, alpha=0.7, color="lightcoral")
    ax2.set_title("Word Count Distribution")
    ax2.set_xlabel("Word Count")
    ax2.set_ylabel("Frequency")
    ax2.axvline(
        data["word_count"].mean(),
        color="red",
        linestyle="--",
        label=f'Mean: {data["word_count"].mean():.0f}',
    )
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return data


def multi_label_analysis(data):
    """Analyze multi-label characteristics and correlations"""
    label_columns = data.columns.difference(["text", "text_length", "word_count"])

    # Number of labels per sample
    data["label_count"] = data[label_columns].sum(axis=1)
    print("\nLabels per sample distribution:")
    label_count_dist = data["label_count"].value_counts().sort_index()
    print(label_count_dist)

    # Plot label count distribution
    plt.figure(figsize=(10, 5))
    label_count_dist.plot(kind="bar", color="lightgreen", alpha=0.7)
    plt.title("Distribution of Number of Labels per Sample")
    plt.xlabel("Number of Labels")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Label co-occurrence correlation matrix
    if len(label_columns) > 1:
        correlation_matrix = data[label_columns].corr()

        plt.figure(figsize=(12, 10))
        mask = correlation_matrix == 1.0  # Hide diagonal
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            fmt=".2f",
            square=True,
            mask=mask,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Label Co-occurrence Correlation Matrix")
        plt.tight_layout()
        plt.show()

        # Find most correlated label pairs
        corr_pairs = []
        for i in range(len(label_columns)):
            for j in range(i + 1, len(label_columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.1:  # Only show moderate+ correlations
                    corr_pairs.append((label_columns[i], label_columns[j], corr_val))

        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        print("\nTop correlated label pairs:")
        for label1, label2, corr in corr_pairs[:10]:
            print(f"{label1} <-> {label2}: {corr:.3f}")

    return data


def text_quality_assessment(data, text_column="text"):
    """Assess text quality and identify potential issues"""
    print("\n=== TEXT QUALITY ASSESSMENT ===")

    # Empty or very short texts
    short_texts = data[data[text_column].str.len() < 5]
    print(
        f"Texts with less than 5 characters: {len(short_texts)} ({len(short_texts)/len(data)*100:.2f}%)"
    )

    # Very long texts (potential outliers)
    long_texts = data[data[text_column].str.len() > 500]
    print(
        f"Texts with more than 500 characters: {len(long_texts)} ({len(long_texts)/len(data)*100:.2f}%)"
    )

    # Duplicate texts
    duplicates = data[data[text_column].duplicated()]
    print(f"Duplicate texts: {len(duplicates)} ({len(duplicates)/len(data)*100:.2f}%)")

    # Most common texts
    common_texts = data[text_column].value_counts().head(5)
    print("\nMost common texts (top 5):")
    for i, (text, count) in enumerate(common_texts.items(), 1):
        print(f"{i}. '{text[:60]}...' (appears {count} times)")

    # Check for potentially problematic patterns
    empty_after_strip = data[text_column].str.strip().str.len() == 0
    print(f"Texts that are only whitespace: {empty_after_strip.sum()}")

    # Single character/word texts
    single_word = data[text_column].str.split().str.len() == 1
    print(
        f"Single word texts: {single_word.sum()} ({single_word.sum()/len(data)*100:.2f}%)"
    )


def emotion_text_analysis(data, sample_size=3):
    """Show sample texts for each emotion category"""
    label_columns = data.columns.difference(
        ["text", "text_length", "word_count", "label_count"]
    )

    print("\n=== SAMPLE TEXTS FOR EACH EMOTION ===")
    for emotion in label_columns:
        emotion_count = data[emotion].sum()
        if emotion_count > 0:
            emotion_texts = data[data[emotion] == 1]["text"].sample(
                min(sample_size, emotion_count)
            )
            print(f"\n{emotion.upper()} ({emotion_count} samples):")
            for i, text in enumerate(emotion_texts, 1):
                display_text = text[:80] + "..." if len(text) > 80 else text
                print(f"  {i}. {display_text}")
        else:
            print(f"\n{emotion.upper()}: No samples found")


def advanced_text_statistics(data, text_column="text"):
    """Calculate advanced text statistics"""
    print("\n=== ADVANCED TEXT STATISTICS ===")

    # Vocabulary size
    all_words = " ".join(data[text_column]).split()
    unique_words = set(all_words)
    print(f"Total words: {len(all_words):,}")
    print(f"Unique words: {len(unique_words):,}")
    print(f"Vocabulary richness: {len(unique_words)/len(all_words):.4f}")

    # Average sentence length (assuming sentences end with . ! ?)
    import re

    sentences = []
    for text in data[text_column]:
        if isinstance(text, str):
            sent_count = len(re.split(r"[.!?]+", text))
            if sent_count > 0:
                sentences.append(len(text.split()) / sent_count)

    if sentences:
        avg_words_per_sentence = sum(sentences) / len(sentences)
        print(f"Average words per sentence: {avg_words_per_sentence:.2f}")

    # Text complexity (average word length)
    word_lengths = []
    for text in data[text_column]:
        if isinstance(text, str):
            words = text.split()
            word_lengths.extend([len(word) for word in words])

    if word_lengths:
        avg_word_length = sum(word_lengths) / len(word_lengths)
        print(f"Average word length: {avg_word_length:.2f} characters")


def data_quality_assessment(data, text_column="text"):
    print("\n" + "=" * 60)
    print("📊 DATA QUALITY ASSESSMENT REPORT")
    print("=" * 60)

    # 1. Missing values
    print("\n❌ Missing Values:")
    missing = data.isnull().sum()
    missing_pct = (missing / len(data)) * 100
    missing_info = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    print(missing_info[missing_info["Missing Count"] > 0])
    if missing_info["Missing Count"].sum() == 0:
        print("✅ No missing values found")

    # 2. Duplicate texts
    if text_column in data.columns:
        duplicate_count = data[text_column].duplicated().sum()
        print(
            f"\n📑 Duplicate texts: {duplicate_count} ({duplicate_count/len(data)*100:.2f}%)"
        )
    else:
        print("\n⚠️ No text column found for duplicate check")

    # 3. Text quality
    if text_column in data.columns:
        text_len = data[text_column].str.len()
        short_texts = (text_len < 5).sum()
        long_texts = (text_len > 500).sum()
        whitespace_only = (data[text_column].str.strip().str.len() == 0).sum()

        print("\n🔤 Text Quality:")
        print(f" - Texts < 5 chars: {short_texts} ({short_texts/len(data)*100:.2f}%)")
        print(f" - Texts > 500 chars: {long_texts} ({long_texts/len(data)*100:.2f}%)")
        print(f" - Texts only whitespace: {whitespace_only}")

    # 4. Label distribution
    label_columns = data.columns.difference(
        [text_column, "text_length", "word_count", "label_count"]
    )
    if len(label_columns) > 0:
        label_sums = data[label_columns].sum().sort_values(ascending=False)
        print("\n🏷️ Label Distribution (top 10):")
        print(label_sums.head(10))

        # Labels per sample
        labels_per_sample = data[label_columns].sum(axis=1)
        print("\n📌 Multi-label stats:")
        print(f" - Avg labels per sample: {labels_per_sample.mean():.2f}")
        print(f" - Samples with 0 labels: {(labels_per_sample == 0).sum()}")
        print(f" - Samples with 1 label: {(labels_per_sample == 1).sum()}")
        print(f" - Samples with >1 label: {(labels_per_sample > 1).sum()}")
    else:
        print("\n⚠️ No label columns found for distribution check")

    # 5. Vocabulary richness
    if text_column in data.columns:
        all_words = " ".join(data[text_column].astype(str)).split()
        unique_words = set(all_words)
        vocab_richness = len(unique_words) / len(all_words) if len(all_words) > 0 else 0
        print("\n📚 Vocabulary statistics:")
        print(f" - Total words: {len(all_words):,}")
        print(f" - Unique words: {len(unique_words):,}")
        print(f" - Vocabulary richness: {vocab_richness:.4f}")

    print("\n✅ Data Quality Assessment Complete")
    print("=" * 60)


if __name__ == "__main__":
    print("🚀 Starting Comprehensive EDA Pipeline...")

    # Step 1: Load raw data
    raw_data = load_raw_data()

    # Step 2: EDA BEFORE cleaning (to understand what needs to be cleaned)
    emotion_cols = eda_before_cleaning(raw_data)

    # Step 3: Clean the data
    cleaned_data = clean_data(raw_data)

    # Step 4: Preprocess text
    cleaned_data = preprocess_text(cleaned_data)

    # Step 5: EDA AFTER cleaning (to understand the final dataset)
    eda_after_cleaning(cleaned_data, emotion_cols)

    # Step 6: Advanced EDA - Text Length Analysis
    print("\n" + "=" * 50)
    print("📏 TEXT LENGTH ANALYSIS")
    print("=" * 50)
    cleaned_data = text_length_analysis(cleaned_data)

    # Step 7: Advanced EDA - Multi-Label Analysis
    print("\n" + "=" * 50)
    print("🏷️  MULTI-LABEL ANALYSIS")
    print("=" * 50)
    cleaned_data = multi_label_analysis(cleaned_data)

    # Step 8: Advanced EDA - Text Quality Assessment
    print("\n" + "=" * 50)
    print("✨ TEXT QUALITY ASSESSMENT")
    print("=" * 50)
    text_quality_assessment(cleaned_data)

    # Step 9: Advanced EDA - Advanced Text Statistics
    print("\n" + "=" * 50)
    print("📊 ADVANCED TEXT STATISTICS")
    print("=" * 50)
    advanced_text_statistics(cleaned_data)

    # Step 10: Advanced EDA - Emotion-specific Text Analysis
    print("\n" + "=" * 50)
    print("🎭 EMOTION-SPECIFIC TEXT SAMPLES")
    print("=" * 50)
    emotion_text_analysis(cleaned_data)

    # Step 11: Save cleaned data
    save_cleaned_data(cleaned_data)

    print("\n" + "=" * 60)
    print("🎉 COMPREHENSIVE EDA AND DATA CLEANING COMPLETE!")
    print("=" * 60)

    # Final summary
    print("\n📋 FINAL DATASET SUMMARY:")
    print(f"   • Total samples: {len(cleaned_data):,}")
    print(f"   • Total features: {len(cleaned_data.columns)}")
    print("   • Text column: 'text'")
    print(f"   • Label columns: {len(emotion_cols)}")
    print(
        f"   • Average text length: {cleaned_data['text'].str.len().mean():.1f} chars"
    )
    print(
        f"   • Average word count: {cleaned_data['text'].str.split().str.len().mean():.1f} words"
    )
    if "label_count" in cleaned_data.columns:
        print(
            f"   • Average labels per sample: {cleaned_data['label_count'].mean():.2f}"
        )

    print("\n💾 Cleaned dataset saved as 'Dataset/goemotions_cleaned.csv'")
    print("🚀 Ready for machine learning modeling!")


# ========== MACHINE LEARNING PIPELINE ========== #
print("\n" + "=" * 60)
print("🤖 MULTI-LABEL EMOTION CLASSIFICATION PIPELINE")
print("=" * 60)



# Ensure stopwords are downloaded
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))


def preprocess_text_ml(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove non-letters
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)


# Dùng cleaned_data từ pipeline EDA
df = cleaned_data.copy()
df["clean_text"] = df["text"].astype(str).apply(preprocess_text_ml)

# ========== PREPARE LABELS ========== #
emotion_cols = [
    c
    for c in df.columns
    if c not in ["text", "clean_text", "text_length", "word_count", "label_count"]
]
mlb = MultiLabelBinarizer()

# Convert từ multi-hot (0/1 per column) → list nhãn
labels = df[emotion_cols].values
labels_list = [list(np.array(emotion_cols)[row.astype(bool)]) for row in labels]
Y = mlb.fit_transform(labels_list)

# ========== SPLIT DATA ========== #
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], Y, test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=20000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ========== DEFINE MODELS ========== #
models = {
    "Logistic Regression": OneVsRestClassifier(LogisticRegression(max_iter=300)),
    "Naive Bayes": OneVsRestClassifier(MultinomialNB()),
    "Linear SVM": OneVsRestClassifier(LinearSVC()),
    "Random Forest": OneVsRestClassifier(RandomForestClassifier(n_estimators=100)),
}

# ========== TRAIN & EVALUATE ========== #
for name, model in models.items():
    print("=" * 60)
    print(f"🚀 Training {name}...")
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    print(f"✅ Results for {name}:")
    print(
        classification_report(
            y_test, y_pred, target_names=mlb.classes_, zero_division=0
        )
    )
