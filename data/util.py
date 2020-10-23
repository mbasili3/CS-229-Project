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
    df = pd.read_csv(csv_path)
    # inputs = np.asarray([a for a in df['x'].to_numpy()])
    # labels = df['y'].to_numpy()
    data_head = df.head()
    print(data_head)

    # return inputs, labels


load_dataset("full_data_one_hot.csv")
# data = pd.read_csv('full_data.csv', sep=',')
