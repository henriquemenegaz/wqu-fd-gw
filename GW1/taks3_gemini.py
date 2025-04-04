# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# We will use yfinance to attempt to download real data.
# If this fails, we'll need to manually provide data or use placeholders.
import yfinance as yf
import datetime

# --- Configuration ---
# Set a random seed for reproducibility
np.random.seed(42)
# Define the number of variables/securities
n_variables = 5
# Define the number of observations for simulated data
n_observations = 252 # Roughly number of trading days in a year

# --- Part a: Generate 5 Uncorrelated Gaussian Random Variables ---
print("--- Part a: Generating Simulated Uncorrelated Data ---")
# Define parameters for simulated yield changes
sim_mean = 0
sim_std_dev = 0.001 # Small standard deviation (e.g., 0.1% or 10 basis points)

# Generate uncorrelated data (each column generated independently)
simulated_yield_changes = pd.DataFrame(
    np.random.normal(sim_mean, sim_std_dev, size=(n_observations, n_variables)),
    columns=[f'SimVar_{i+1}' for i in range(n_variables)]
)
print("Simulated Uncorrelated Data (First 5 rows):")
print(simulated_yield_changes.head())
print("\nCorrelation Matrix of Simulated Data (should be near identity):")
print(simulated_yield_changes.corr())

# --- Part b: Run PCA on Simulated Data (using Correlation Matrix approach) ---
print("\n--- Part b: Running PCA on Simulated Data ---")
# To use the correlation matrix, we first standardize the data (mean=0, std dev=1)
scaler_sim = StandardScaler()
simulated_scaled = scaler_sim.fit_transform(simulated_yield_changes)

# Apply PCA. sklearn's PCA works on the covariance of the input.
# Applying it to standardized data is equivalent to PCA on the correlation matrix.
pca_sim = PCA(n_components=n_variables)
pca_sim.fit(simulated_scaled)

# Explained variance ratio for simulated data
explained_variance_ratio_sim = pca_sim.explained_variance_ratio_
print("PCA completed for simulated data.")

# --- Part d: Produce Scree Plot (Simulated Data) ---
print("\n--- Part d: Generating Scree Plot for Simulated Data ---")
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_variables + 1), explained_variance_ratio_sim, 'o-', linewidth=2, label='Simulated Data')
plt.title('Scree Plot - Simulated Uncorrelated Yield Changes')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.xticks(range(1, n_variables + 1))
plt.ylim(bottom=0)
plt.grid(True, linestyle='--', alpha=0.7)
# Add percentage labels
for i, v in enumerate(explained_variance_ratio_sim):
    plt.text(i + 1.1, v, f"{v*100:.1f}%", fontsize=9)
plt.show()
print("Scree plot for simulated data generated.")


# --- Part e: Collect Real Government Security Yield Data ---
print("\n--- Part e: Collecting Real Government Yield Data ---")
# Define tickers for US Treasury yields (using FRED tickers via yfinance if possible)
# FRED tickers: DGS3MO (3-Month), DGS2 (2-Year), DGS5 (5-Year), DGS10 (10-Year), DGS30 (30-Year)
# Note: yfinance syntax for FRED might require adding '.FRED' (e.g., 'DGS5.FRED'),
# but often just the ticker works if yfinance recognizes it as a FRED source.
fred_tickers_yf = ['DGS3MO', 'DGS2', 'DGS5', 'DGS10', 'DGS30']
used_tickers = []
real_data_list = []

# Define time period (approx. last 6 months from today, April 4, 2025)
end_date = datetime.date(2025, 4, 4)
start_date = end_date - datetime.timedelta(days=180)

print(f"Attempting to download data for: {fred_tickers_yf} from {start_date} to {end_date}")

for ticker in fred_tickers_yf:
    try:
        # Append .F to hint FRED source if direct ticker fails sometimes
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)['Adj Close']
        if data.empty:
             print(f"No data for {ticker}, trying {ticker}.F")
             data = yf.download(ticker + '.F', start=start_date, end=end_date, progress=False)['Adj Close']

        if not data.empty:
            # FRED data often has NaNs for non-trading days, need to forward fill
            data = data.ffill()
            real_data_list.append(data)
            used_tickers.append(ticker)
            print(f"Successfully downloaded {ticker}")
        else:
            print(f"Failed to download {ticker} or {ticker}.F - Data empty.")

    except Exception as e:
        print(f"Error downloading {ticker}: {e}")

# Check if we got enough data
if len(real_data_list) >= 3: # Need at least 3 for somewhat meaningful PCA
    # Combine into a single DataFrame and align dates
    real_yield_levels = pd.concat(real_data_list, axis=1)
    real_yield_levels.columns = used_tickers # Name columns with tickers used
    real_yield_levels = real_yield_levels.dropna(how='all') # Drop rows where ALL data is missing
    real_yield_levels = real_yield_levels.ffill() # Forward fill gaps
    real_yield_levels = real_yield_levels.dropna() # Drop any rows with remaining NaNs (important!)

    print("\nReal Data Yield Levels (Last 5 rows):")
    print(real_yield_levels.tail())

    # --- Part f: Compute Daily Yield Changes ---
    print("\n--- Part f: Computing Daily Yield Changes ---")
    real_yield_changes = real_yield_levels.diff().dropna() # Calculate diff and remove first NaN row
    print("Real Data Yield Changes (Last 5 rows):")
    print(real_yield_changes.tail())
    print("\nCorrelation Matrix of Real Yield Changes (should show high correlation):")
    print(real_yield_changes.corr())

    # Update n_variables based on successfully downloaded data
    n_real_variables = real_yield_changes.shape[1]

    # --- Part g: Re-run PCA on Real Data (using Correlation Matrix approach) ---
    print("\n--- Part g: Running PCA on Real Data ---")
    # Standardize the real yield changes
    scaler_real = StandardScaler()
    real_scaled = scaler_real.fit_transform(real_yield_changes)

    # Apply PCA
    pca_real = PCA(n_components=n_real_variables) # Use actual number of variables
    pca_real.fit(real_scaled)

    # Explained variance ratio for real data
    explained_variance_ratio_real = pca_real.explained_variance_ratio_
    print("PCA completed for real data.")

    # --- Part i: Produce Scree Plot (Real Data) ---
    print("\n--- Part i: Generating Scree Plot for Real Data ---")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_real_variables + 1), explained_variance_ratio_real, 'o-', linewidth=2, color='red', label='Real Data')
    plt.title(f'Scree Plot - Real Government Yield Changes ({n_real_variables} Maturities)')
    plt.xlabel('Principal Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.xticks(range(1, n_real_variables + 1))
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', alpha=0.7)
    # Add percentage labels
    for i, v in enumerate(explained_variance_ratio_real):
        plt.text(i + 1.1, v, f"{v*100:.1f}%", fontsize=9)
    plt.show()
    print("Scree plot for real data generated.")

else:
    # --- Placeholder if data download fails ---
    print("\nERROR: Could not download sufficient real data points. Cannot proceed with real data PCA.")
    # Set flags to indicate real data analysis was skipped
    pca_real = None
    explained_variance_ratio_real = None
    n_real_variables = 0


# --- Markdown Explanations (Parts c, h, j) ---

print("\n\n--- MARKDOWN EXPLANATIONS ---")

# --- Part c: Variance Explained (Simulated Data) ---
print("\n## Part c: Variance Explained by Components (Simulated Uncorrelated Data)")
print("\n```markdown")
print("Principal Component Analysis (PCA) was performed on the 5 simulated, uncorrelated Gaussian variables representing yield changes. Since the variables were generated independently and standardized before PCA (equivalent to using the correlation matrix), there is no inherent structure or dominant direction of variance in the data for PCA to exploit. As a result, the total variance is distributed relatively evenly among the principal components. Each component captures a portion of the variance that is roughly equal to 1 divided by the number of variables (1/5 = 20%).")
print("\nSpecifically:")
# Use f-string to dynamically insert results
print(f"* **Component 1** explains approximately **{explained_variance_ratio_sim[0]*100:.2f}%** of the variance.")
print(f"* **Component 2** explains approximately **{explained_variance_ratio_sim[1]*100:.2f}%** of the variance.")
print(f"* **Component 3** explains approximately **{explained_variance_ratio_sim[2]*100:.2f}%** of the variance.")
print(f"* The remaining components ({len(explained_variance_ratio_sim)-3}) explain similarly small proportions.")
print("This near-equal distribution confirms the lack of correlation in the input data; no single component summarizes significantly more information than the others.")
print("```")


# --- Part h: Variance Explained (Real Data) ---
print("\n## Part h: Variance Explained by Components (Real Government Yield Data)")
print("\n```markdown")
if pca_real: # Check if PCA was run on real data
    print(f"PCA was performed on the daily changes of {n_real_variables} real government bond yields, again using the correlation matrix approach (by standardizing the yield changes). Unlike the simulated data, real government bond yields across different maturities are known to be highly correlated – they tend to move together. PCA effectively captures this shared movement.")
    print("\nThe variances explained by each component demonstrate this clearly:")
    # Use f-string to dynamically insert results
    print(f"* **Component 1** explains a very large majority of the variance, approximately **{explained_variance_ratio_real[0]*100:.2f}%**. This component typically represents the 'level' factor – parallel shifts up or down in the entire yield curve.")
    if n_real_variables > 1:
        print(f"* **Component 2** explains a much smaller, but often significant, portion, around **{explained_variance_ratio_real[1]*100:.2f}%**. This component often corresponds to the 'slope' factor – changes in the steepness of the yield curve (e.g., long rates moving more or less than short rates).")
    if n_real_variables > 2:
        print(f"* **Component 3** explains an even smaller amount, roughly **{explained_variance_ratio_real[2]*100:.2f}%**. This is sometimes associated with the 'curvature' or 'butterfly' factor – how the middle part of the curve moves relative to the ends.")
    if n_real_variables > 3:
         print(f"* Subsequent components explain progressively very little variance, often considered noise.")
    print(f"\nThe rapid decrease in explained variance after the first few components shows that the dimensionality of the yield curve movements can be effectively reduced. The first {min(3, n_real_variables)} components capture most of the systematic behavior.")
else:
    print("Real data analysis was skipped due to insufficient data downloaded.")
print("```")


# --- Part j: Scree Plot Comparison ---
print("\n## Part j: Comparison of Scree Plots")
print("\n```markdown")
print("Comparing the two scree plots reveals the fundamental difference between analyzing uncorrelated data and correlated financial data like yield changes:")
print("\n1.  **Simulated Uncorrelated Data Scree Plot:**")
print(f"* **Shape:** Relatively flat. The line connecting the variance proportions for each component has a gentle slope.")
print(f"* **Interpretation:** This flatness indicates that each principal component contributes roughly equally to explaining the total variance (around {100/n_variables:.1f}% each in this 5-variable case). This is characteristic of datasets where the original variables are independent or uncorrelated. Dimensionality reduction via PCA is not particularly effective here, as all original dimensions contribute almost equally.")
print("\n2.  **Real Government Yield Data Scree Plot:**")
if pca_real:
    print(f"* **Shape:** Steep decline initially, followed by an 'elbow' and a flattening tail. The first point (PC1) is significantly higher than the second, the second is noticeably higher than the third, and subsequent points are very low and close together.")
    print(f"* **Interpretation:** The steep drop after PC1 (explaining ~{explained_variance_ratio_real[0]*100:.1f}%) shows that a single factor (the 'level' shift) accounts for the vast majority of the daily yield movements. The subsequent, smaller contributions of PC2 (~{explained_variance_ratio_real[1]*100:.1f}%) and PC3 (~{explained_variance_ratio_real[2]*100:.1f}%) capture secondary, but still potentially important, patterns ('slope' and 'curvature'). The rapid leveling off ('scree') indicates that components beyond the first few capture very little systematic variance (likely noise). This shape is typical for highly correlated data and demonstrates that PCA can effectively summarize the main patterns in the yield curve movements using far fewer dimensions than the original number of yields.")
else:
    print("* The scree plot for real data could not be generated due to data download issues.")
    print("* However, typically, this plot would show a steep decline initially, followed by an 'elbow' and a flattening tail, indicating that the first few components capture most of the variance due to high correlation between yield changes.")

print("\n**In summary:** The contrast between the flat scree plot (uncorrelated data) and the steep 'elbow' scree plot (real yield data) visually demonstrates how PCA identifies and leverages correlation structure to achieve dimensionality reduction and extract meaningful underlying factors (like level, slope, curvature for yields).")
print("```")