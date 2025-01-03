import pandas as pd

def load_data(url):
    """Loads the dataset from the given URL."""
    df = pd.read_csv(url)
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    return df

def preprocess_data(df):
    """Groups by year and interpolates missing values."""
    annual_avg = df.groupby('Year')['Mean'].mean().reset_index()
    annual_avg['Mean'] = annual_avg['Mean'].interpolate(method='linear')
    return annual_avg
