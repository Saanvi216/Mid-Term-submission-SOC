import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(
    'data/processed/AAPL_features.csv',
    index_col="Date",
    parse_dates=True
)

print('Shape:', df.shape)
print(df.head())
