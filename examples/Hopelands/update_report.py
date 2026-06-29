import pandas as pd
import os
import re
import glob

def main():
    output_dir = 'examples/Hopelands/output'
    report_path = 'examples/Hopelands/Final_Analysis_Report.md'

    if not os.path.exists(report_path):
        print(f"Error: {report_path} not found.")
        return

    with open(report_path, 'r') as f:
        report_content = f.read()

    # Find NSE from 1_...out
    nse = "N/A"
    out_files = glob.glob(os.path.join(output_dir, '1_DE-MCMC_NSE_Hopelands_*.out'))
    if out_files:
        with open(out_files[0], 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                nse = lines[1].strip()

    # 1. Update Executive Summary
    # Matches "NSE) of **...**" or "NSE of **...**"
    report_content = re.sub(r"NSE\)? of \*\*[^*]+\*\*", f"NSE) of **{nse}**", report_content)

    # 2. Update Performance Metrics Table
    metrics_file = os.path.join(output_dir, 'goodness_of_fit_calibration_DE-MCMC_NSE_Hopelands.csv')

    if os.path.exists(metrics_file):
        print(f"Updating performance metrics from {metrics_file}...")
        df_metrics = pd.read_csv(metrics_file)
        metrics_dict = {row['Metric']: row['Value'] for _, row in df_metrics.iterrows()}

        new_metrics_table = "| Metric | Value |\n|--------|-------|\n"
        new_metrics_table += f"| NSE    | {nse} |\n"
        new_metrics_table += f"| R²     | {metrics_dict.get('R2', 0.0):.4f} |\n"
        new_metrics_table += f"| RMSE   | {metrics_dict.get('RMSE', 0.0):.3f}  |\n"
        new_metrics_table += f"| MAE    | {metrics_dict.get('MAE', 0.0):.3f}  |\n"

        table_pattern = r"### 3.2. Performance Metrics\n\| Metric \| Value \|\n\|--------\|-------\|\n(?:\|.*?\|.*?\|\n)*"
        report_content = re.sub(table_pattern, f"### 3.2. Performance Metrics\n{new_metrics_table}", report_content, flags=re.DOTALL)

    # 3. Update Parameter Table (with CIs)
    sig_file = os.path.join(output_dir, 'parameter_significance_DE-MCMC_Hopelands.csv')
    if os.path.exists(sig_file):
        print(f"Updating parameter significance table from {sig_file}...")
        df_sig = pd.read_csv(sig_file)
        md_table = "| Parameter | Mean | 95% CI Lower | 95% CI Upper | Significant |\n"
        md_table += "|-----------|------|--------------|--------------|-------------|\n"
        for _, row in df_sig.iterrows():
            md_table += f"| {row['Parameter']} | {row['Mean']:.4f} | {row['95%_CI_Lower']:.4f} | {row['95%_CI_Upper']:.4f} | {row['Significantly_Diff_From_Zero']} |\n"

        section_marker = "### 3.3. Parameter Significance and Uncertainty"
        pattern = f"{re.escape(section_marker)}\\n.*?(?=\\n\\s*!\\[)"
        report_content = re.sub(pattern, f"{section_marker}\n\n{md_table}\n", report_content, flags=re.DOTALL)

    # Remove excessive blank lines
    report_content = re.sub(r'\n{3,}', '\n\n', report_content)

    with open(report_path, 'w') as f:
        f.write(report_content)
    print("Updated Final_Analysis_Report.md")

    # 4. Export Synthetic Data
    sim_files = glob.glob(os.path.join(output_dir, '2_DE-MCMC_NSE_Hopelands_*c_1d.csv'))
    if sim_files:
        sim_file = sim_files[0]
        print(f"Exporting synthetic data from {sim_file}...")
        df_sim = pd.read_csv(sim_file)
        df_sim = df_sim[df_sim['Year'] != -999].copy()
        df_sim['Date'] = pd.to_datetime(df_sim[['Year', 'Month', 'Day']])
        df_synthetic = df_sim[df_sim['Twat_mod'] != -999.0][['Date', 'Twat_mod']]
        synthetic_out = os.path.join(output_dir, 'Synthetic_Water_Temperature_Hopelands.csv')
        df_synthetic.to_csv(synthetic_out, index=False)
        print(f"Exported synthetic water temperature to {synthetic_out}")

if __name__ == '__main__':
    main()
