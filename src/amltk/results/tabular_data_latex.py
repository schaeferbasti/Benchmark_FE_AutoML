import pandas as pd


def format_value(val):
    """
    Format values to 4 decimal places if they are in the "mean ± std" format.
    Otherwise, return the value as is (e.g., for "Failed").
    """
    if isinstance(val, str) and "±" in val:
        try:
            # Split the value into mean and std dev
            mean, std = val.split(" ± ")
            # Format both to 4 decimal places
            formatted_val = f"{float(mean):.4f} $\\pm$ {float(std):.4f}"
            return formatted_val
        except ValueError:
            return val  # In case splitting or conversion fails, return the original string
    else:
        return val  # Return non-numeric values (like "Failed") unchanged


def highlight_max_in_row(row):
    """
    Helper function to apply LaTeX bold formatting to the max value in a row,
    while formatting numbers to 4 decimal places.
    """
    # Handle cases where some values may be NaN or the string "Failed"
    valid_values = row[1:].apply(lambda x: float(x.split(' ')[0]) if isinstance(x, str) and "±" in x else float('nan'))
    max_value = valid_values.max()

    # Apply LaTeX formatting and round values to 4 decimal places
    return [
        f"\\textbf{{{format_value(val)}}}" if valid == max_value and not pd.isna(valid)
        else format_value(val)
        for val, valid in zip(row[1:], valid_values)
    ]


def format_latex_row(dataset, row):
    """
    Formats a single row into LaTeX style.
    """
    # Ensure the dataset name is a string and remove the number if present
    dataset_name = str(dataset).split(' ', 1)[-1]  # Splits by space and takes the part after the number
    return f"{dataset_name} & " + " & ".join(row) + " \\\\"


def generate_latex_table(df):
    """
    Generates a LaTeX table from the given DataFrame.
    """
    # Apply the highlighting function to each row (skipping the first column)
    df_highlighted = df.apply(highlight_max_in_row, axis=1)

    # Begin LaTeX table structure
    latex_str_1 = r"""\begin{landscape}
\begin{table}
\tiny
\begin{tabular}{l""" + "c" * (len(df.columns) - 1) + "}\n\\toprule\n"
    latex_str_2 = r"""
\begin{tabular}{l""" + "c" * (len(df.columns) - 1) + "}\n\\toprule\n"

    # Add column headers (excluding the first column)
    latex_str_1 += choose_cols("Dataset & " + " & ".join(df.columns[1:]) + " \\\\\n\\midrule\n", 1)
    latex_str_2 += choose_cols("Dataset & " + " & ".join(df.columns[1:]) + " \\\\\n\\midrule\n", 2)
    # Add each row of the DataFrame
    for dataset, row in zip(df.iloc[:,0], df_highlighted.values):
        latex_row = format_latex_row(dataset, row) + "\n"
        latex_row_1 = choose_cols(latex_row, 1)
        latex_str_1 += latex_row_1
        latex_row_2 = choose_cols(latex_row, 2)
        latex_str_2 += latex_row_2
    # End LaTeX table structure
    latex_str_1 += r"""\bottomrule
\end{tabular}"""
    latex_str_2 += r"""\bottomrule
\end{tabular}
\label{tab:small-benchmark}
\caption{Overview of the comparison between the performance of the \AMLTK{} pipeline with feature engineered version of the original datasets vs. the original raw dataset.}
\end{table}
\end{landscape}"""


    return latex_str_1, latex_str_2

def choose_cols(row, x):
    row_list = row.split("&")
    if x == 1:
        row = "&".join(row_list[:int(len(row_list) / 2)]) + "&" + row_list[-1]
        return row
    if x == 2:
        row = row_list[0] + "&" + "&".join(row_list[int(len(row_list) / 2):])
        return row


# Read the parquet file
df = pd.read_parquet('tabular_data.parquet')

# Generate the LaTeX formatted table
latex_table_1, latex_table_2 = generate_latex_table(df)

# Print the LaTeX formatted table
print(latex_table_1)
print(latex_table_2)
