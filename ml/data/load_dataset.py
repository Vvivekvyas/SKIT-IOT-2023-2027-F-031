from pathlib import Path
import pandas as pd


def load_csv(file_path):
    """
    Load a single CSV file.

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected CSV file, got: {file_path.suffix}"
        )

    df = pd.read_csv(file_path)

    return df


def load_dataset_folder(folder_path, dataset_name=None):
    """
    Load and combine all CSV files from a folder
    and its subfolders.

    Parameters
    ----------
    folder_path : str or Path
        Root folder containing CSV files.

    dataset_name : str, optional
        Name of the dataset.

    Returns
    -------
    pd.DataFrame
        Combined dataset.
    """

    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder_path}"
        )

    # rglob searches the folder and all subfolders
    csv_files = sorted(folder_path.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {folder_path}"
        )

    dataframes = []

    for file in csv_files:

        print(f"Loading: {file}")

        df = pd.read_csv(file)

        # Keep track of the original file
        df["Source_File"] = str(file.relative_to(folder_path))

        # Keep track of the dataset
        if dataset_name is not None:
            df["Dataset"] = dataset_name

        dataframes.append(df)

    combined_df = pd.concat(
        dataframes,
        ignore_index=True
    )

    return combined_df


def validate_dataset(df):
    """
    Perform basic validation on a loaded dataset.
    """

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nDuplicate rows:", df.duplicated().sum())

    print(
        "\nMissing values:",
        df.isnull().sum().sum()
    )

    print(
        "Infinite values:",
        df.select_dtypes(include="number")
          .isin([float("inf"), float("-inf")])
          .sum()
          .sum()
    )

    print("\nData types:")
    print(df.dtypes.value_counts())

    return True