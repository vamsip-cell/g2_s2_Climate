
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# GENERATE SAMPLE DATA
# =========================================================

def generate_sample_data():
    """Generate one scenario for PDF/CDF plots"""
    size = 64
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    
    # Create mixed scenario
    np.random.seed(42)
    
    # Stratiform component
    R_strat = 5 + 2 * np.random.randn(size, size)
    R_strat = gaussian_filter(R_strat, sigma=5)
    R_strat = np.maximum(R_strat, 0.1)
    
    # Convective component
    R_conv = np.zeros((size, size))
    for _ in range(8):
        cx, cy = np.random.randint(10, 54, 2)
        intensity = np.random.uniform(20, 50)
        r2 = (x - cx)**2 + (y - cy)**2
        cell = intensity * np.exp(-r2 / 50)
        R_conv += cell
    R_conv = np.maximum(R_conv, 0.1)
    
    # Mixed truth
    R_truth = 0.3 * R_strat + 0.7 * R_conv
    R_truth = np.maximum(R_truth, 0.1)
    
    # Create forecast with errors
    R_forecast = gaussian_filter(R_truth, sigma=1.5)
    R_forecast = np.roll(R_forecast, shift=1, axis=1)
    R_forecast = R_forecast * 0.95
    R_forecast = np.maximum(R_forecast, 0.1)
    
    return R_truth, R_forecast

# =========================================================
# CLEAN PDF AND CDF PLOTS
# =========================================================

def plot_clean_pdf_cdf(R_truth, R_forecast, scenario_name="Mixed"):
    """
    Generate clean PDF and CDF plots - no extra labels, no annotations
    """
    print(f"Generating PDF and CDF plots for: {scenario_name}")
    
    # Flatten arrays
    fcst_flat = R_forecast.flatten()
    truth_flat = R_truth.flatten()
    
    # Remove zeros
    fcst_rain = fcst_flat[fcst_flat > 0.1]
    truth_rain = truth_flat[truth_flat > 0.1]
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # =========================================================
    # PDF PLOT (left)
    # =========================================================
    if len(truth_rain) > 0 and len(fcst_rain) > 0:
        # Log-spaced bins
        bins = np.logspace(-1, 2, 30)
        
        # Plot histograms
        ax1.hist(truth_rain, bins=bins, alpha=0.7, density=True, 
                label='Observed', color='blue', edgecolor='black', linewidth=1)
        ax1.hist(fcst_rain, bins=bins, alpha=0.7, density=True,
                label='Forecast', color='red', edgecolor='black', linewidth=1)
        
        # Log scales
        ax1.set_xscale('log')
        ax1.set_yscale('log')
    
    # Labels and title
    ax1.set_xlabel('Rainfall Intensity (mm/h)', fontsize=11)
    ax1.set_ylabel('Probability Density', fontsize=11)
    ax1.set_title(f'PDF - {scenario_name}', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    # =========================================================
    # CDF PLOT (right)
    # =========================================================
    if len(truth_rain) > 0 and len(fcst_rain) > 0:
        # Sort values
        truth_sorted = np.sort(truth_rain)
        fcst_sorted = np.sort(fcst_rain)
        
        # CDF values
        truth_cdf = np.arange(1, len(truth_sorted)+1) / len(truth_sorted)
        fcst_cdf = np.arange(1, len(fcst_sorted)+1) / len(fcst_sorted)
        
        # Plot
        ax2.plot(truth_sorted, truth_cdf, 'b-', linewidth=2, label='Observed')
        ax2.plot(fcst_sorted, fcst_cdf, 'r--', linewidth=2, label='Forecast')
        ax2.set_xscale('log')
    
    # Labels and title
    ax2.set_xlabel('Rainfall Intensity (mm/h)', fontsize=11)
    ax2.set_ylabel('Cumulative Probability', fontsize=11)
    ax2.set_title(f'CDF - {scenario_name}', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.2, linestyle='--')
    
    # Main title
    plt.suptitle('Deterministic S-PROG: Distribution Analysis', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('pdf_cdf_clean.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    # Generate data
    print("Generating sample rainfall data...")
    R_truth, R_forecast = generate_sample_data()
    
    # Generate clean plots
    plot_clean_pdf_cdf(R_truth, R_forecast, "Mixed Weather")
