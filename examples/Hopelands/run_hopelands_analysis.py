import subprocess
import sys
import os

def run_command(command, description):
    print(f"--- {description} ---")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error during {description}")
        sys.exit(result.returncode)

def main():
    # Ensure we are in the right environment
    # 1. Preprocessing
    run_command("PYTHONPATH=. python examples/Hopelands/preprocess.py", "Data Preprocessing and Merging")

    # 2. Main Analysis
    run_command("python -m pyair2stream.main --config examples/Hopelands/config.yaml", "Model Calibration and Analysis")

    # 3. Report Update and Synthetic Data Export
    run_command("PYTHONPATH=. python examples/Hopelands/update_report.py", "Updating Analysis Report and Exporting Synthetic Data")

    print("\nAnalysis complete. Results are in examples/Hopelands/output/")
    print("Final report can be found at examples/Hopelands/README.md")

if __name__ == '__main__':
    main()
