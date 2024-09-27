#%% imports
import numpy as np
import json 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from pathlib import Path
from typing import List, Tuple

#%%
def main():
    #%%
    black_out_dir = "black_output"
    teal_out_dir = "teal_output"
    metrics = [
        "WearTime(days)",
        "TotalSteps",
        "Cadence95th(steps/min)",
    ]

    def extract_metrics(dir):
        # glob all the json files
        outputs = list(Path(dir).rglob("*.json"))
        df = {metric : [] for metric in metrics}
        for output in outputs:
            with open(output, "r") as f:
                accel = json.load(f)
                for metric in metrics:
                    df[metric].append(accel[metric])
        return pd.DataFrame(df)


    # %%
    black_df = extract_metrics(black_out_dir)
    teal_df = extract_metrics(teal_out_dir)

    #  %%
    print(f"black median total steps: {black_df['TotalSteps'].median()}")
    print(f"teal median total steps: {teal_df['TotalSteps'].median()}")

    print(f"black median cadence 95: {black_df['Cadence95th(steps/min)'].median()}")
    print(f"teal median cadence 95: {teal_df['Cadence95th(steps/min)'].median()}")

    print(f"black median weartime: {black_df['WearTime(days)'].median()}")
    print(f"teal median weartime: {teal_df['WearTime(days)'].median()}")

    #%% set up fonts
    plt.rcParams["font.family"] = "Baskerville"


    #%% Scatter plot of total steps against cadence
    fig, ax = plt.subplots(figsize=(6,4))
    ax.scatter(black_df["TotalSteps"], black_df["Cadence95th(steps/min)"], alpha=0.5, color="black", s=10)
    ax.scatter(teal_df["TotalSteps"], teal_df["Cadence95th(steps/min)"], alpha=0.5, color="teal", s=10)
    ax.set_title("Spring in the step vs. total steps")
    ax.set_xlabel("Total steps")
    ax.set_ylabel("Cadence\n95th percentile\n(steps/min)", rotation=0, ha="right")
    plt.show()


    #%% Boxplot of total steps

    vecs=[ black_df["TotalSteps"].to_list(), teal_df["TotalSteps"].to_list()]

    fig, ax = tufte_boxplot(
        vecs=vecs,
        x_labels=["Black", "Teal"],
        colours=["black", "teal"],
        y_label="Total steps",
        title="Minimalist box-plot of total steps",
        show_points=True,
    )
    plt.show()

    #%% Boxplot of cadence 95

    vecs=[ black_df["Cadence95th(steps/min)"].to_list(), teal_df["Cadence95th(steps/min)"].to_list()]

    fig, ax = tufte_boxplot(
        vecs=vecs,
        x_labels=["Black", "Teal"],
        colours=["black", "teal"],
        y_label="Cadence\n95th percentile\n(steps/min)",
        title="Minimalist box-plot of cadence 95th percentile",
        show_points=True,
    )
    plt.show()

    #%% Boxplot of weartime
    vecs= np.concatenate([ black_df["WearTime(days)"].to_list(), teal_df["WearTime(days)"].to_list()])

    fig, ax = tufte_boxplot(
        vecs=vecs,
        y_label="Wear time (days)",
        title="Minimalist box-plot wear time",
        show_points=True,
    )
    plt.show()



# %%
def calculate_statistics(vec, calc_outliers=True):
    """Calculate the quartiles, median, and outliers of a dataset."""
    vec = np.array(vec)
    q1 = np.percentile(vec, 25)
    median = np.percentile(vec, 50)
    q3 = np.percentile(vec, 75)
    iqr = q3 - q1
    if calc_outliers:
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        min_val = np.min(vec[vec >= lower_bound])
        max_val = np.max(vec[vec <= upper_bound])
        outliers = vec[(vec < lower_bound) | (vec > upper_bound)]
    else:
        min_val = np.min(vec)
        max_val = np.max(vec)
        outliers = np.array([])
    
    return min_val, q1, median, q3, max_val, outliers



def tufte_boxlines(
    ax, i, min_val, q1, median, q3, max_val, colour, offset, gap, 
):
    """
    Looks like:
    |
     |
     |
    |
    """
    ax.add_line(Line2D([i, i], [min_val, q1], color=colour))
    ax.add_line(Line2D([i + offset, i + offset], [q1, median-gap], color=colour))
    ax.add_line(Line2D([i + offset, i + offset], [median+gap, q3], color=colour))
    ax.add_line(Line2D([i, i], [q3, max_val], color=colour))

def tufte_boxdot(
        ax, i, min_val, q1, median, q3, max_val, colour, median_dot_size, 
):
    """
    Looks like:
    |
    .
    |
    """
    ax.add_line(Line2D([i, i], [min_val, q1], color=colour))
    ax.scatter(i, median, color=colour, s=median_dot_size)
    ax.add_line(Line2D([i, i], [q3, max_val], color=colour))



def tufte_boxplot(
        vecs: np.ndarray | List | List[np.ndarray],
        title: str = "",
        y_label: str = "",
        x_labels: List[str] = "",
        colours: str | List[str] = "grey",
        # outliers
        calc_outliers: bool = True,
        outlier_size: int = 10,
        outlier_marker: str = ".",
        # scatter points
        show_points: bool = False,
        point_marker: str = "o",
        alpha: float = 0.5,
        scatter_offset: float = -0.2,
        scatter_sd: float = 0.02,
        scatter_seed: int = 42,
        point_size: int = 10,
        # label median
        label_median: bool = True,
        median_font_size: int = 10,
        median_as_dot: bool = True,
        # for line based plot
        gap: float = 0.02,
        offset: float = 0.02,
        # for dot as median
        median_dot_size: int = 10,
        # Spacing
        y_margin: float = 0.1, # space after the first, last point proportional to the length of the plot
        x_margin: float = 0.5,
        figsize: Tuple[int] = (6,4),
        spines_off: List[str]=["top", "right", "bottom", "left"],
):
    """
    Plots boxplots as per Tufte's minimalist redesign.
    
    Args:
    - vecs: vector of length P or vectors of shape CxP.
    - title: optional title of the plot.
    - y_label: y-label of the plot.
    - x_labels: x-label if vecs is a vector or C labels if vecs is C vectors.
    - calc_outliers: define outliers as points 1.5*IQR and show them separately.
    - show_points: plot the underlying points with some jitter to the left of the box plot.
    - label_median: label the median value next to the median gap.
    """
    if not isinstance(vecs[0], (np.ndarray, list)):
        flat_vec = vecs
        vecs = [vecs]  # convert to list of 1 vector for consistency
        x_labels = [x_labels] if isinstance(x_labels, str) else x_labels
    else:
        flat_vec = np.concatenate(vecs)
        
    fig, ax = plt.subplots(figsize=figsize)

    # Calc. stats of all vecs
    min_val, q1, median, q3, max_val, outliers = calculate_statistics(flat_vec, calc_outliers=calc_outliers)
    all_min_val = np.min(list(outliers) + [min_val])
    all_max_val = np.max(list(outliers) + [max_val])
    max_min = all_max_val - all_min_val
    offset = offset
    gap = gap * max_min / 2

    # Colour shape
    if isinstance(colours, str):
        colours = [colours] * len(vecs)
    
    # Set up rng for scattering points
    if show_points:
        rng = np.random.default_rng(scatter_seed)

    for i, (vec, colour) in enumerate(zip(vecs, colours)):
        # Calculate the statistics
        min_val, q1, median, q3, max_val, outliers = calculate_statistics(vec, calc_outliers=calc_outliers)

        # Draw the Tufte-style minimalist boxplot
        if median_as_dot:
            tufte_boxdot(ax=ax, i=i, min_val=min_val, q1=q1, median=median, q3=q3, max_val=max_val, colour=colour, median_dot_size=median_dot_size)
        else:
            tufte_boxlines(ax=ax, i=i, min_val=min_val, q1=q1, median=median, q3=q3, max_val=max_val, colour=colour, offset=offset, gap=gap)
        
        # Label the median
        if label_median:
            ax.text(i + offset, median, f'{median:.2f}', verticalalignment='center', fontsize=median_font_size)

        # Plot the data points (optional)
        if show_points:
            jittered_x = rng.normal(loc=i+scatter_offset, scale=scatter_sd, size=len(vec))
            ax.scatter(x=jittered_x, y=vec, alpha=alpha, color=colour, s=point_size, marker=point_marker, zorder=2)

        elif calc_outliers and len(outliers) > 0: # Optionally plot outliers
            ax.scatter(x=[i] * len(outliers), y=outliers, color=colour, s=outlier_size, marker=outlier_marker, zorder=3)

    # Set the x and y limits
    y_margin = max_min * y_margin
    ax.set_ylim(all_min_val-y_margin,all_max_val+y_margin)
    x_margin = len(vecs) * x_margin
    ax.set_xlim(0 - x_margin, i + x_margin)
    # Add the x-labels
    ax.set_xticks(range(len(vecs)))
    ax.set_xticklabels(x_labels)
    # ... y label
    ax.set_ylabel(y_label, rotation=0, ha="right")
    # Title  
    ax.set_title(title)

    # Remove the box around the figure
    for spine_off in spines_off:
        ax.spines[spine_off].set_visible(False)

    return fig, ax


# %%
