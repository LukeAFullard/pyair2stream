with open("pyair2stream/post_processing.py", "r") as f:
    lines = f.read()

search_block = """
        ax_res.set_xlabel('Time')
        ax_res.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
        fig.autofmt_xdate()

        plt.tight_layout()
        pdf_path = os.path.join(data.folder, f"{output_name}_{data.runmode}_{data.fun_obj}_{data.station}.pdf")
        png_path = os.path.join(data.folder, f"{output_name}_{data.runmode}_{data.fun_obj}_{data.station}.png")
        plt.savefig(pdf_path, dpi=300)
        plt.savefig(png_path, dpi=300)
        plt.close()
"""

replace_block = """
        ax_res.set_xlabel('Time')
        ax_res.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
        fig.autofmt_xdate()

        plt.tight_layout()
        pdf_path = os.path.join(data.folder, f"{output_name}_{data.runmode}_{data.fun_obj}_{data.station}.pdf")
        png_path = os.path.join(data.folder, f"{output_name}_{data.runmode}_{data.fun_obj}_{data.station}.png")
        plt.savefig(pdf_path, dpi=300)
        plt.savefig(png_path, dpi=300)
        plt.close()

        # Generate Residual Diagnostics Plot
        # Drop NaNs from residuals
        res_clean = residuals.dropna()
        if len(res_clean) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Histogram
            axes[0].hist(res_clean, bins=30, edgecolor='black', alpha=0.7)
            axes[0].set_title(f'Residuals Histogram ({output_name})')
            axes[0].set_xlabel('Residual [°C]')
            axes[0].set_ylabel('Frequency')

            # Q-Q plot
            import scipy.stats as stats
            stats.probplot(res_clean, dist="norm", plot=axes[1])
            axes[1].set_title(f'Normal Q-Q Plot ({output_name})')

            # Autocorrelation (ACF)
            pd.plotting.autocorrelation_plot(res_clean, ax=axes[2])
            axes[2].set_title(f'Autocorrelation ({output_name})')
            axes[2].set_xlim([0, min(50, len(res_clean))]) # Limit x-axis to first 50 lags for readability

            plt.tight_layout()
            diag_pdf = os.path.join(data.folder, f"residual_diagnostics_{output_name}_{data.runmode}_{data.fun_obj}_{data.station}.pdf")
            diag_png = os.path.join(data.folder, f"residual_diagnostics_{output_name}_{data.runmode}_{data.fun_obj}_{data.station}.png")
            plt.savefig(diag_pdf, dpi=300)
            plt.savefig(diag_png, dpi=300)
            plt.close()
"""

lines = lines.replace(search_block, replace_block)

with open("pyair2stream/post_processing.py", "w") as f:
    f.write(lines)
