# Heartrate Anomaly Detection using Matrix Profiling

This project implements anomaly detection on heartrate time-series data using the **Matrix Profile** technique, powered by the [STUMPY](https://stumpy.readthedocs.io/) library. It reads input data, computes matrix profiles, identifies anomalies, visualizes results, and exports findings for further analysis.

## Features
- **Matrix Profile Computation**: Detects unusual patterns in time-series data using STUMPY.
- **Anomaly Detection**: Identifies timestamps where anomalies occur based on a configurable threshold.
- **Data Normalization**: Optionally normalizes data to improve detection consistency.
- **Interactive Plotting**: Visualizes time series and anomalies with Plotly.
- **Results Export**: Saves computed profiles, scores, and anomalies to a pickle file.

## Requirements
- Python 3.8+
- Required packages:
  - `numpy`
  - `pandas`
  - `stumpy`
  - `plotly`
  - `pickle` (standard library)
  - `functools` (standard library)
  - `datetime` (standard library)

Install dependencies using:
```bash
pip install numpy pandas stumpy plotly
```

## Usage
1. Place your heartrate data as a CSV file in the `data` directory (default expects `heartrate_personal_reduced.csv`).
2. Run the script:
```bash
python src/matrix-profiling.py
```
3. The script will:
   - Read the input data
   - Compute matrix profiles
   - Detect anomalies using the specified threshold
   - Display plots for each column
   - Export results to `mp_exported.pkl`

## Configuration
You can adjust parameters directly in the script:
- `window_size`: Sliding window size for matrix profile computation
- `smooth_n`: Rolling window size for smoothing
- `normalize_data`: Boolean to enable/disable normalization
- `plot_results`: Boolean to enable/disable plotting
- `threshold`: Float to set the anomaly detection threshold

## Output
- **Printed Output**: Displays top scores and detected anomalies.
- **Exported Pickle File**: Contains:
  - `mp_dists`: Matrix profile distances per column
  - `df_scores`: Scores and rankings
  - `anomalies`: DataFrame with detected anomalies

## Example
The script outputs something like:

```
            score  rank
heartrate    0.12   1.0

   timestamp     column    value  index  mp_score
0 2023-05-01   heartrate   85.0     150     0.95
1 2023-05-02   heartrate   120.0    320     1.10
```

## License
This project is provided as-is for research and educational purposes.
