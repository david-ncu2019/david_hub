#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

sys.path.append(r"D:\VENV_PYTHON\py3_8\Scripts")
sys.path.append(r"D:\VINHTRUONG\DAVID_GIT\psi_workflow")

from my_packages import *
from PSI_Toolbox import *
from signal_toolbox import *


# In[2]:


def extract_well_name_from_filepath(filepath):
    """
    Extracts the well name from the given file path.

    Args:
        filepath (str): The file path from which to extract the well name.

    Returns:
        str: The extracted well name.
    """
    return os.path.basename(filepath).split(".")[0]


# | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def differ_to_ref(series):
    """
    Differencing the series to the reference point and converts to centimeters.

    Args:
        series (pandas.Series): The series to be differenced.

    Returns:
        pandas.Series: The differenced series.
    """
    return (series - series.iloc[0]) * 100


# | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def get_minmax_compaction(df):
    max_df = np.ceil(df.max().max()) + 1
    min_df = np.floor(df.min().min()) - 1

    max_compaction = min_df * -1
    min_compaction = max_df * -1
    return [min_compaction, max_compaction]


# | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def load_compaction_data(filepath):
    """
    Loads and preprocesses compaction data from a specified file.

    Args:
        filepath (str): File path of the compaction data.

    Returns:
        tuple: A tuple containing the DataFrame of compaction data, an array mapping column names to depths, and the depth of the deepest ring.
    """
    # Load the data from the file
    original_data = pd.read_pickle(filepath)
    original_data.index = pd.to_datetime(original_data.index)

    # Sort the index if necessary
    sorted_index = original_data.index.sort_values()
    df = original_data.loc[sorted_index, :].copy()

    # Transpose the DataFrame for easier processing
    df = df.transpose()

    # Extract depth information from column names
    colname_to_depth = np.array(
        [eval(colname.split("_")[-1].split(" ")[0]) for colname in df.index]
    )
    deepest_ring = max(colname_to_depth)

    # Apply the differ_to_ref function or any other preprocessing steps
    df = df.apply(differ_to_ref, axis="columns")

    return df, colname_to_depth, deepest_ring


# | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# In[3]:


# | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def initialize_figure(width_cm, height_cm):
    """Initialize the figure and axis for plotting."""
    cm_to_inches = 1 / 2.54
    fig = plt.figure(figsize=(width_cm * cm_to_inches, height_cm * cm_to_inches))
    ax = fig.add_subplot(111)
    return fig, ax


# | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def plot_column_data(ax, df, colname_to_depth, count):
    """
    Plots compaction data for each column in a DataFrame.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to plot the data.
        df (pandas.DataFrame): The DataFrame containing compaction data.
        colname_to_depth (numpy.array): Array of depth values corresponding to columns.
        count (int): Counter used for selecting color from the colormap.

    This function iterates over each column in the DataFrame, representing different
    time points, and plots the compaction data on the provided axis.
    """
    cmap = plt.get_cmap("jet")

    deepest_ring_latest_day_min = df.iloc[-1, :].min()
    shallowest_ring_oldest_day_max = df.iloc[0, :].max()

    for col, label in zip(df.columns, df.columns.strftime("%Y/%m/%d")):
        color = cmap(float(count) / len(df.columns))

        select_array = df[col]
        select_array[select_array < deepest_ring_latest_day_min] = np.nan
        select_array[select_array > shallowest_ring_oldest_day_max] = np.nan

        ax.plot(
            select_array * -1,
            colname_to_depth[::-1] * -1,
            label=label,
            marker="o",
            linestyle="-",
            color=color,
        )
        count += 1


# | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def SetPlotLabelsAndTitle(ax, select_well, translations):
    """
    Sets labels and title for a plot.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to set labels and title.
        select_well (str): The well name used for the title.
        translations (dict): A dictionary for translating well names.

    The function sets the y-label, x-label, and title for the plot and also adjusts
    the position of the x-axis ticks and labels to the top of the plot.
    """
    ax.set_ylabel("Depth (m)", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_xlabel("Compaction (cm)", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title(translations[select_well].upper(), fontsize=20, fontweight="bold", pad=20)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")


# | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def SetAxisTicksAndLocators(ax, ymajor_interval=50, yminor_interval=25):
    """
    Sets the y-axis major and minor tick locators for the plot.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to set the locators.
        ymajor_interval (int): Interval for major ticks on the y-axis.
        yminor_interval (int): Interval for minor ticks on the y-axis.

    This function configures the y-axis with major and minor tick intervals
    to improve the readability of the plot.
    """
    y_major_loc = plticker.MultipleLocator(base=ymajor_interval)
    y_minor_loc = plticker.MultipleLocator(base=yminor_interval)
    ax.yaxis.set_major_locator(y_major_loc)
    ax.yaxis.set_minor_locator(y_minor_loc)


# | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def SetPlotLimits(ax, df, deepest_ring):
    """
    Sets the limits for the plot.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to set the limits.
        df (pandas.DataFrame): DataFrame containing the data.
        deepest_ring (int): The deepest ring measurement in the data.

    Determines and sets the x and y-axis limits based on the compaction data
    and the deepest ring measurement.
    """
    left_bound, right_bound = get_minmax_compaction(df)
    interval = np.diff(ax.get_xticks())[0]
    ax.set_xlim(-interval, right_bound)
    ax.set_ylim(bottom=-max(deepest_ring, 300), top=20)


# | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def CustomizePlotAppearance(ax):
    """
    Customizes the appearance of the plot.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to customize the appearance.

    This function customizes the y-axis tick labels, sets tick parameters, adds grid
    lines, and hides certain plot spines for a cleaner and more readable plot appearance.
    """
    current_ytick_labels = [label.get_text() for label in ax.get_yticklabels()]
    new_ytick_labels = [ele.replace("−", "") for ele in current_ytick_labels]
    ax.set_yticklabels(new_ytick_labels)
    ax.tick_params(axis="y", which="major", labelsize=14, direction="in", length=8)
    ax.tick_params(axis="y", which="minor", labelsize=14, direction="in", length=5)
    ax.tick_params(axis="x", which="major", labelsize=14, direction="in", length=8)
    ax.tick_params(axis="x", which="minor", labelsize=14, direction="in", length=5)
    ax.grid(which="major", axis="x", color="grey", alpha=0.3, linestyle="--")
    ax.grid(which="major", axis="y", color="grey", alpha=0.3, linestyle="--")
    for side in ["right", "bottom"]:
        ax.spines[side].set_visible(False)


# | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def AddColorBar(fig, ax, df):
    """
    Adds a color bar to the plot.

    Args:
        fig (matplotlib.figure.Figure): The figure to which the color bar is added.
        ax (matplotlib.axes.Axes): The axis on which the color bar is associated.
        df (pandas.DataFrame): DataFrame containing the data.

    This function adds a vertical color bar to the plot, indicating different time
    points in the data. It uses the 'jet' colormap to represent the progression of time.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    modified_datetime = df.columns.strftime("%Y/%m/%d")
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=1))
    cbaxes = inset_axes(
        ax,
        width="20%",
        height="2%",
        loc="lower left",
        bbox_to_anchor=(0.75, 0.05, 0.1, 20),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(sm, cax=cbaxes, orientation="vertical")
    cbar.ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    _len = len(modified_datetime)
    datestring_list = [
        modified_datetime[X][:]
        for X in [-1, int(_len * 0.8), int(_len * 0.6), int(_len * 0.4), int(_len * 0.2), 0]
    ]
    cbar.ax.set_yticklabels(datestring_list[::-1], fontsize=14, rotation=0)


# | - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# In[4]:


folder = r"E:\SUBSIDENCE_PROJECT_DATA\地陷資料整理\地陷井\監測井_資料清理結果"
files = glob(os.path.join(folder, "*.xz"))

well_code = [os.path.basename(file).split(".")[0] for file in files]

well_meta = pd.read_excel(r"E:\SUBSIDENCE_PROJECT_DATA\地陷資料整理\地陷井\select_well.xlsx")

available_wells = [well for well in well_code if well in well_meta.iloc[:, 15].to_list()]

translations = {
    "二崙": "Erlun",
    "僑義": "Qiaoyi",
    "元長": "Yuanchang",
    "光復": "Guangfu",
    "內寮": "Neiliao",
    "北辰": "Beichen",
    "南光": "Nanguang",
    "嘉興": "Jiaxing",
    "土庫": "Tuku",
    "安南": "Annan",
    "宏崙": "Honglun",
    "宜梧": "Yiwu",
    "客厝": "Kecuo",
    "崙豐_新": "Xin Lunfeng",
    "建陽": "Jianyang",
    "拯民": "Zhengmin",
    "新生": "Xinsheng",
    "新興": "Xinxing",
    "新街": "Xinjie",
    "東光": "Dongguang",
    "海豐": "Haifeng",
    "湖南": "Hunan",
    "溪州": "Xizhou",
    "燦林": "Canlin",
    "秀潭": "Xiutan",
    "竹塘": "Zhutang",
    "興華": "Xinghua",
    "舊庄": "Jiuzhuang",
    "虎尾": "Huwei",
    "西港": "Xigang",
    "豐安": "Fengan",
    "豐榮": "Fengrong",
    "金湖_新": "Xin Jinhu",
    "鎮南": "Zhennan",
    "龍岩": "Longyan",
}


# In[5]:


for select_well in available_wells[:2]:
    # filepath = r"E:\SUBSIDENCE_PROJECT_DATA\地陷資料整理\\地陷井\監測井_資料清理結果\土庫.xz"
    filepath = [element for element in files if select_well in element][0]
    df, colname_to_depth, deepest_ring = load_compaction_data(filepath)

    fig, ax = initialize_figure(width_cm=29.7 * 0.5, height_cm=21)

    plot_column_data(ax=ax, df=df, colname_to_depth=colname_to_depth, count=0)

    # Apply the new subroutines
    SetPlotLabelsAndTitle(ax, select_well, translations)
    SetAxisTicksAndLocators(ax)
    SetPlotLimits(ax, df, deepest_ring)
    CustomizePlotAppearance(ax)
    AddColorBar(fig, ax, df)

    plt.show()


# In[ ]:




