with open("pyair2stream/post_processing.py", "r") as f:
    lines = f.read()

import re

search_block = """
    # 1. Dotty plots
    if os.path.exists(file_0):
        # Read the parameter tracking CSV
        df_0 = pd.read_csv(file_0)
"""
replace_block = """
    # 1a. Convergence Plots
    if os.path.exists(file_0):
        df_opt = pd.read_csv(file_0)

        if 'NSE' in df_opt.columns and 'R2' in df_opt.columns and 'MAE' in df_opt.columns:
            fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

            # Cumulative best for NSE and R2 (maximize)
            df_opt['cummax_nse'] = df_opt['NSE'].cummax()
            df_opt['cummax_r2'] = df_opt['R2'].cummax()

            # Cumulative best for MAE (minimize)
            df_opt['cummin_mae'] = df_opt['MAE'].cummin()

            x_vals = range(1, len(df_opt) + 1)

            axes[0].plot(x_vals, df_opt['NSE'], '.', color='lightgray', alpha=0.5, label='Evaluation NSE')
            axes[0].plot(x_vals, df_opt['cummax_nse'], '-', color='blue', linewidth=2, label='Cumulative Best NSE')
            axes[0].set_ylabel('NSE')
            axes[0].set_title('Optimization Convergence')
            axes[0].legend()

            axes[1].plot(x_vals, df_opt['R2'], '.', color='lightgray', alpha=0.5, label='Evaluation R2')
            axes[1].plot(x_vals, df_opt['cummax_r2'], '-', color='green', linewidth=2, label='Cumulative Best R2')
            axes[1].set_ylabel('R2')
            axes[1].legend()

            axes[2].plot(x_vals, df_opt['MAE'], '.', color='lightgray', alpha=0.5, label='Evaluation MAE')
            axes[2].plot(x_vals, df_opt['cummin_mae'], '-', color='red', linewidth=2, label='Cumulative Best MAE')
            axes[2].set_ylabel('MAE [°C]')
            axes[2].set_xlabel('Evaluation Number')
            axes[2].legend()

            plt.tight_layout()
            conv_pdf = os.path.join(data.folder, f"convergence_{data.runmode}_{data.fun_obj}_{data.station}.pdf")
            conv_png = os.path.join(data.folder, f"convergence_{data.runmode}_{data.fun_obj}_{data.station}.png")
            plt.savefig(conv_pdf, dpi=300)
            plt.savefig(conv_png, dpi=300)
            plt.close()

    # 1b. Dotty plots
    if os.path.exists(file_0):
        # Read the parameter tracking CSV
        df_0 = pd.read_csv(file_0)
"""
lines = lines.replace(search_block, replace_block)

with open("pyair2stream/post_processing.py", "w") as f:
    f.write(lines)
