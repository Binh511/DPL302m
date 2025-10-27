import pandas as pd

class DataEvaluator:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = pd.read_csv(data_path)


    def print_info(self):
        print("Dataset shape:", self.data.shape)
        print("Dataset columns:", self.data.columns.tolist())

    def max_length(self, text_column):
        """
        return max token
        """
        return self.data[text_column].apply(lambda x: len(str(x).split())).max()


# ===--- Main ---===
Deva = DataEvaluator('Database/cleaned_goemotions.csv')

max_token = Deva.max_length('text')
print("Max token length in 'text' column:", max_token)

