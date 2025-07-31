import numpy as np
import pandas as pd
import os
import stumpy
import plotly.express as px
import functools

import pickle

from datetime import datetime

def time_log(f):
  '''
  Decorator to log the execution time of functions.
  '''
  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    tick = datetime.now()
    results = f(*args, **kwargs)
    tock = datetime.now()
    print(f"{f.__name__} function took {tock - tick}")
    return results
  wrapper.unwrapped = f
  return wrapper

def normalize(df) -> pd.DataFrame:
  '''
  Normalize dataframe columns to the [0,1] range.

  Parameters:
      df (pd.DataFrame): Input dataframe.

  Returns:
      pd.DataFrame: Normalized dataframe.
  '''
  return (df - df.min())/(df.max() - df.min())

@time_log
def run_matrix_profile_anomaly_detection(df, window_size=96, smooth_n=10, normalize_data=True, plot_results=True, threshold: float = None):
    '''
    Compute matrix profile anomaly detection.

    Parameters:
        df (pd.DataFrame): Input time series data.
        window_size (int): Size of the sliding window.
        smooth_n (int): Smoothing factor for profile distances.
        normalize_data (bool): Whether to normalize input data.
        plot_results (bool): Whether to plot results using Plotly.
        threshold (float, optional): Threshold to detect anomalies.

    Returns:
        tuple: (mp_dists (dict), df_scores (pd.DataFrame), anomalies (pd.DataFrame))
    '''
    scores = {}
    mp_dists = {}
    anomalies = []

    if len(df.shape) <= 1:
        df = df.to_frame()
    else:
        pass

    if normalize_data:
        df = (normalize(df).dropna())
    else:
        pass

    for column in df.columns:
        mp = stumpy.stump(df[column].astype("float"), m=window_size, ignore_trivial=True, normalize=False)

        mp_dist = mp[:, 0]

        df_mp_dist = pd.DataFrame(mp_dist).rolling(smooth_n).mean()
        scores[column] = df_mp_dist[0].quantile(0.995) - df_mp_dist[0].quantile(0.75)

        nan_value_count = np.empty(len(df[column]) - len(mp_dist))
        nan_value_count.fill(np.nan)
        mp_dist = np.concatenate((nan_value_count, mp_dist.astype(float)))

        mp_dists[column] = mp_dist

        # Per inserire le anomalie in un df
        if threshold is not None:
          anomaly_indices = np.where(mp_dist > threshold)[0]
          for idx in anomaly_indices:
            time = df.index[idx]
            anomalies.append({'timestamp': time, 'column': column, 'value': df[column].iloc[idx], 'index': idx, 'mp_score': mp_dist[idx]})

    if anomalies:
      anomalies = pd.DataFrame(anomalies)
    else:
      anomalies = pd.DataFrame(columns=['timestamp', 'column', 'value', 'index', 'mp_score'])

    df_scores = pd.DataFrame.from_dict(scores, orient='index', columns=['score'])
    df_scores['rank'] = df_scores['score'].rank(ascending=False)

    if plot_results:

        pd.options.plotting.backend = "plotly"

        for column in df_scores.sort_values('rank').index:
            score = round(df_scores.loc[column]['score'], 3)
            rank = df_scores.loc[column]['rank']
            df_tmp = df[[column]].assign(MatrixProfileScore = mp_dists[column])
            fig = df_tmp.plot(template="simple_white", title=f"{column}")
            fig.show()
        else:
            pass

    return mp_dists, df_scores.sort_values('rank'), anomalies

def pickle_export(mp_dists, df_scores, anomalies, pkl_filename):
  '''
  Export computed results to a pickle file.

  Parameters:
      mp_dists (dict): Matrix profile distances.
      df_scores (pd.DataFrame): Scores dataframe.
      anomalies (pd.DataFrame): Anomalies dataframe.
      pkl_filename (str): Output pickle filename.
  '''
  with open(pkl_filename, 'wb') as f:
        pickle.dump({'mp_dists': mp_dists, 'df_scores': df_scores, 'anomalies': anomalies}, f)

@time_log
def read(filepath: str) -> pd.DataFrame:
  '''
  Read CSV data into a pandas DataFrame, parse timestamps, and set index.

  Parameters:
      filepath (str): Path to the CSV file.

  Returns:
      pd.DataFrame: Loaded and sorted dataframe with timestamp index.
  '''
  df = (pd.read_csv(filepath, parse_dates=True).assign(timestamp = lambda x : pd.to_datetime(x.timestamp)).set_index(['timestamp']).sort_index())
  return df

data = read('../data/heartrate_personal_reduced.csv')
matrix_profile_distances, scores_df, anomaly_df = (run_matrix_profile_anomaly_detection(data, window_size=(30), smooth_n=(150), normalize_data=True, plot_results=True, threshold=0.8))

pickle_export(matrix_profile_distances, scores_df, anomaly_df, 'mp_exported.pkl')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

print(scores_df.head(5))
print('\n')
print('\n')
print(anomaly_df.head(50))