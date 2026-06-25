import pandas as pd
import matplotlib.pyplot as plt

def main():
    # We will read both results
    fortran_cal_df = pd.read_csv('fortran/src/output_8/2_PSO_NSE_DAV_2327_cc_1d.out', sep=r'\s+', header=None, skiprows=1)
    python_cal_df = pd.read_csv('examples/validation/Switzerland/output_8/2_DE_NSE_DAV_cc_1d.csv')

    fortran_val_df = pd.read_csv('fortran/src/output_8/3_PSO_NSE_DAV_2327_cv_1d.out', sep=r'\s+', header=None, skiprows=1)
    python_val_df = pd.read_csv('examples/validation/Switzerland/output_8/3_DE_NSE_DAV_cv_1d.csv')

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Calibration Plot (Twat_mod comparison)
    axes[0].plot(fortran_cal_df.iloc[:, 5].values, label='Fortran', alpha=0.7)
    axes[0].plot(python_cal_df['Twat_mod'].values, label='Python', alpha=0.7, linestyle='--')
    axes[0].set_title('Calibration - Simulated Water Temperature')
    axes[0].legend()

    # Validation Plot
    axes[1].plot(fortran_val_df.iloc[:, 5].values, label='Fortran', alpha=0.7)
    axes[1].plot(python_val_df['Twat_mod'].values, label='Python', alpha=0.7, linestyle='--')
    axes[1].set_title('Validation - Simulated Water Temperature')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('examples/validation/Switzerland/comparison.png')
    print("Comparison plot saved to examples/validation/Switzerland/comparison.png")

if __name__ == '__main__':
    main()
