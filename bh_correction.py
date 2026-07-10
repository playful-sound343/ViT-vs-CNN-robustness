import pandas as pd
from statsmodels.stats.multitest import multipletests

# I extracted these directly from your screenshot
data = {
    "Experiment": [
        "Clean accuracy", 
        "Dropout B01", "Dropout B02", "Dropout B03", "Dropout B04", "Dropout B05", "Dropout B06",
        "Noise x1.0", "Noise x5.0", 
        "Haze OD=0.2", "Haze OD=0.5", 
        "Visible (B02-B04)", "Red Edge (B05-B07)", "NIR (B08+B08A)", "SWIR (B11-B12)", "NDVI (Red+NIR)", "NDWI (Green+NIR)"
    ],
    "p_value_raw": [
        0.045, 
        0.041, 0.507, 0.906, 0.035, 0.091, 0.001, 
        0.003, 0.003, 
        0.039, 0.005, 
        0.253, 0.002, 0.245, 0.002, 0.038, 0.291
    ]
}

df = pd.DataFrame(data)

# Perform Benjamini-Hochberg FDR correction with an alpha of 0.05
reject, pvals_corrected, _, _ = multipletests(
    df['p_value_raw'], 
    alpha=0.05, 
    method='fdr_bh'
)

# Add results back to the dataframe
df['p_val_BH_corrected'] = pvals_corrected
df['Significant (FDR < 0.05)'] = reject

# Sort by the corrected p-value so the most significant results are at the top
df = df.sort_values(by='p_val_BH_corrected')

print("=== Benjamini-Hochberg Corrected Results ===")
print(df.to_string(index=False))