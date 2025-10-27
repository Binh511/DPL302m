"""
Reason to use ALBERT:
- Sentimental Analysis doesn't not requires a deep model, Shared weights in ALBERT help reduce the model size
- Data's max token length (<128) is not long, and comes in sentences. ALBERT performs well on such data.

"""

import pandas as pd

data_path = 'Database/cleaned_goemotions.csv'
data = pd.read_csv(data_path)


# Data Preprocessing Exploration
from transformers import AlbertTokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')


    

    