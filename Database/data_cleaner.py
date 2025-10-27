"""
For the dataset, we:
- Merge multiple csv into one
- Remove unrelated columns
- No empty rows to remove
- No duplicates to remove (but will check)

For each line of text we:
- Lowercase
- Remove Special Characters
- Remove Stopwords (I will look into if this is necessary later)
- Lemmatize
- Named Entity Recognition (NER) and replace entities with their labels (e.g., PERSON, ORG)
- Misspelled words correction


Variation of this dataset includes:
- Full merged dataset with all emotions (27 columns for emotions + 1 text column)
- Cleaned merged dataset with all emotions (removed unrelated columns, cleaned text)
- Cleaned merged dataset with merged labels (Positive, Negative, Neutral) + cleaned text
- Cleaned merged dataset with example_very_unclear removed + cleaned text but not for "!" and "?"
"""



import pandas as pd


def init_spacy():
    import spacy
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        from spacy.cli import download
        download("en_core_web_lg")
        nlp = spacy.load("en_core_web_lg")
    return nlp

def dataset_cleaning(df):
    unrelated_columns = ['id','author','subreddit','link_id','parent_id','created_utc', 'rater_id']
    for col in unrelated_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Common practices: Check for duplicates and missing values and remove them
    df = df.drop_duplicates()
    df = df.dropna()

    return df


def merged_cleaned(data):
    return data

def merged_cleaned_polarity_labels(data):
    return data

def merged_cleaned_no_unclear(data):
    return data






def text_cleaning(text, nlp):
    # lowercase
    text = text.lower()

    # words like "shouldn't" to "should not"
    # Proper contraction mapping
    contractions = {
        "n't": " not",
        "'re": " are", 
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "I'm": " I am",
        "can't": "can not",
        "won't": "will not",
        "shan't": "shall not"
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove Special Characters
    text = ''.join(char for char in text if char.isalnum() or char.isspace())




    return df
if __name__ == "__main__": 
    # Initializations
    nlp = init_spacy()
    df = pd.read_csv('Database/merged_goemotions.csv')
    df = dataset_cleaning(df)




    df.to_csv('Database/cleaned_goemotions.csv', index=False)
    # Load the dataset

