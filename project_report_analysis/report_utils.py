def split_metric(value):
    value = str(value)

    if "±" in value:
        mean, std = value.split("±")
        return float(mean.strip()), float(std.strip())

    return float(value), None


def get_metric_table(df, metric_name, models):
    table = df[
        (df["Data"] == "SBSAT_STD")
        & (df["Eval Type"] == "test")
        & (df["Model"].isin(models))
    ][["Model", "All"]]

    return table.rename(columns={"All": metric_name})


def add_metric_columns(results_table, metrics):
    table = results_table.copy()

    for metric in metrics:
        table[f"{metric}_mean"] = table[metric].apply(lambda x: split_metric(x)[0])
        table[f"{metric}_std"] = table[metric].apply(lambda x: split_metric(x)[1])

    return table


def plot_metric_points(df, metric):
    import matplotlib.pyplot as plt

    mean_column = f"{metric}_mean"

    plot_df = df[["Model", mean_column]].copy()
    plot_df[mean_column] = plot_df[mean_column].astype(float)

    if metric == "R2":
        plot_df = plot_df.sort_values(mean_column, ascending=False)
        xlabel = "R²"
        title = "Test R² by model"
    else:
        plot_df = plot_df.sort_values(mean_column, ascending=True)
        xlabel = metric
        title = f"Test {metric} by model"

    fig, ax = plt.subplots(figsize=(7, 3.5))

    y_positions = list(range(len(plot_df)))

    ax.scatter(
        plot_df[mean_column],
        y_positions,
        s=80,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["Model"])

    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.set_title(title)

    xmin = plot_df[mean_column].min()
    xmax = plot_df[mean_column].max()

    if metric == "R2":
        padding = max((0 - xmin) * 0.15, 0.02)
        ax.set_xlim(xmin - padding, 0 + padding)
        ax.axvline(0, linestyle="--")
    else:
        padding = max((xmax - xmin) * 0.4, 0.02)
        ax.set_xlim(xmin - padding, xmax + padding)

    for y_position, value in zip(y_positions, plot_df[mean_column]):
        value = float(value)
        label = f"{value:.2f}"

        ax.text(
            value,
            y_position,
            f" {label}",
            va="center",
            ha="left",
        )

    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()