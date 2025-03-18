def generate_latex_table(df, title, label):
    """
    Generates a LaTeX formatted table from a given DataFrame containing model evaluation metrics.

    Parameters:
    df (pd.DataFrame): DataFrame containing columns 'patient_id', 'model', 'mae_mean', 'mae_std', 'rmse_mean', and 'rmse_std'.
    title (str): Title of the LaTeX table.
    label (str): LaTeX label for referencing the table.

    Returns:
    str: A string containing the LaTeX formatted table code.
    
    The function organizes the data by unique patient IDs and models, displaying Mean Absolute Error (MAE) and
    Root Mean Squared Error (RMSE) with their respective standard deviations. It also computes and appends 
    model-wise average MAE and RMSE values at the end of the table.
    """
    models = df['model'].unique()

    # Start LaTeX table
    latex_code = f"""
\\begin{{table}}[h]
    \\centering
    \\caption{{{title}}}
    \\resizebox{{\\columnwidth}}{{!}}{{%
    \\begin{{tabular}}{{c|""" + "cc|" * (len(models) - 1) + "cc}\n"

    # Top rule
    latex_code += "        \\toprule\n"

    # Header row with model names
    model_headers = " & ".join(
        [f"\\multicolumn{{2}}{{c{'|' if i < len(models) - 1 else ''}}}{{\\textbf{{{m}}}}}" for i, m in enumerate(models)]
    )
    latex_code += f"        {{\\textbf{{Patient ID}}}} & {model_headers} \\\\n"

    # Mid rules for metric alignment
    col_ranges = [f"{2 + i * 2}-{3 + i * 2}" for i in range(len(models))]
    latex_code += "        " + " ".join([f"\\cmidrule(lr){{{r}}}" for r in col_ranges]) + "\n"
    latex_code += "         & " + " & ".join(["MAE & RMSE"] * len(models)) + " \\\\n"
    latex_code += "         \\midrule\n"

    # Patient rows
    for patient in df['patient_id'].unique():
        row = [str(patient)]
        for model in models:
            model_data = df[(df['patient_id'] == patient) & (df['model'] == model)]
            if not model_data.empty:
                mae_mean = model_data['mae_mean'].values[0]
                mae_std = model_data['mae_std'].values[0]
                rmse_mean = model_data['rmse_mean'].values[0]
                rmse_std = model_data['rmse_std'].values[0]
                row.append(f"{mae_mean:.2f} $\\pm$ {mae_std:.2f}")
                row.append(f"{rmse_mean:.2f} $\\pm$ {rmse_std:.2f}")
            else:
                row.append("--")  # Placeholder if missing data
                row.append("--")
        latex_code += "         " + " & ".join(row) + " \\\\n"

    # Compute model-wise averages
    latex_code += "         \\midrule\n"
    avg_row = ["\\textbf{Avg}"]

    for model in models:
        model_data = df[df['model'] == model]
        if not model_data.empty:
            avg_mae = model_data['mae_mean'].mean()
            avg_rmse = model_data['rmse_mean'].mean()
            avg_row.append(f"\\textbf{{{avg_mae:.2f}}}")
            avg_row.append(f"\\textbf{{{avg_rmse:.2f}}}")
        else:
            avg_row.append("--")
            avg_row.append("--")

    latex_code += "         " + " & ".join(avg_row) + " \\\\n"

    # Bottom rule and closing
    latex_code += "         \\bottomrule\n"
    latex_code += "    \\end{tabular}%\n"
    latex_code += "    }\n"
    latex_code += f"    \\label{{{label}}}\n"
    latex_code += "\\end{table}\n"

    return latex_code
