import pandas as pd
import numpy as np

def export_for_powerbi():
    df = pd.read_csv('countries of the world.csv', decimal=',')
    df.columns = df.columns.str.strip()
    df['Country'] = df['Country'].str.strip()
    df['Region']  = df['Region'].str.strip()
    
    # Impute missing values
    for col in df.columns:
        if df[col].isnull().sum() == 0: continue
        if col == 'Climate':
            guess = df.groupby('Region')['Climate'].apply(lambda x: x.mode().max())
        else:
            guess = df.groupby('Region')[col].median()
        for region in df['Region'].unique():
            mask = (df[col].isnull()) & (df['Region'] == region)
            df.loc[mask, col] = guess[region]
            
    # Add calculated columns
    df['Total_GDP_B'] = (df['GDP ($ per capita)'] * df['Population'] / 1e9).round(2)
    
    # Export
    df.to_csv('powerbi_data.csv', index=False)
    print("✅ Created 'powerbi_data.csv' - Load this into Power BI!")

if __name__ == "__main__":
    export_for_powerbi()
