with open("pyair2stream/post_processing.py", "r") as f:
    lines = f.read()

search_block = """
    # 2. Time-Series Plots
    def plot_series(file_path, title_prefix, output_name):
"""

replace_block = """
    # 1c. MCMC Parameter Significance & Correlation
    env_file_mcmc = os.path.join(data.folder, f"MCMC_envelopes_{data.station}_{data.series}_{data.time_res}.csv")
    chain_filename = os.path.join(data.folder, f"MCMC_chain_{data.station}_{data.series}_{data.time_res}.csv")
    if os.path.exists(chain_filename):
        chain_df = pd.read_csv(chain_filename)

        # Calculate statistics
        stats = []
        for col in chain_df.columns:
            mean_val = chain_df[col].mean()
            std_val = chain_df[col].std()
            ci_lower = chain_df[col].quantile(0.025)
            ci_upper = chain_df[col].quantile(0.975)
            # A simple significance check: if 0 is not in the 95% CI
            significant = not (ci_lower <= 0 <= ci_upper)
            stats.append({
                'Parameter': col,
                'Mean': mean_val,
                'StdDev': std_val,
                '95%_CI_Lower': ci_lower,
                '95%_CI_Upper': ci_upper,
                'Significantly_Diff_From_Zero': significant
            })

        stats_df = pd.DataFrame(stats)
        sig_file = os.path.join(data.folder, f"parameter_significance_{data.runmode}_{data.station}.csv")
        stats_df.to_csv(sig_file, index=False)
        print(f"Saved parameter significance report to {sig_file}")

        # Parameter Correlation Matrix
        corr_matrix = chain_df.corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)

        # Formatting
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45)
        ax.set_yticklabels(corr_matrix.columns)

        # Add text annotations
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha='center', va='center', color='black' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'white')

        ax.set_title("Parameter Correlation Matrix (MCMC Chain)", pad=20)
        plt.tight_layout()
        corr_pdf = os.path.join(data.folder, f"parameter_correlation_{data.runmode}_{data.station}.pdf")
        corr_png = os.path.join(data.folder, f"parameter_correlation_{data.runmode}_{data.station}.png")
        plt.savefig(corr_pdf, dpi=300)
        plt.savefig(corr_png, dpi=300)
        plt.close()

    # 2. Time-Series Plots
    def plot_series(file_path, title_prefix, output_name):
"""

lines = lines.replace(search_block, replace_block)

with open("pyair2stream/post_processing.py", "w") as f:
    f.write(lines)
