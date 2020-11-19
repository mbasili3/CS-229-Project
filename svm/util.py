import numpy as np
import pandas as pd

####


def load_dataset(csv_path):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    # read in and shuffle data
    df = pd.read_csv(csv_path).sample(frac=1)
    # drop indicator columns for destinations
    cols = [c for c in df.columns if c[:11] !=
            'destination' and c[:14] != 'wind_direction']
    df = df[cols]
    # print(len(cols))
    # print(df.columns.values.tolist())

    inputs = df.drop('time_taxi_out', axis=1).values
    labels = df['time_taxi_out'].values
    return inputs, labels
