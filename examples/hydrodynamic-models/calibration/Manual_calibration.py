"""Manual Calibration.

Manual Calibration to calibrate the model Cross section of the hydraulic
model
"""
import datetime as dt
import os
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Hapi.hm.calibration as RC
import Hapi.hm.river as R
from Hapi.hm.interface import Interface
from Hapi.plot.visualizer import Visualize as V

# %% Paths
"""change directory to the processing folder inside the project folder"""
rpath = "examples/Hydrodynamic-models/test_case"
# rpath = ""
saveto = rpath
tolopogy_file = rpath + "/inputs/1d/topo/"
savepath = rpath + "/results/customized_results/"

xs_file = tolopogy_file + "xs_same_downward-3segment.csv"
river_network = tolopogy_file + "/rivernetwork-3segments.txt"
river_slope = tolopogy_file + "/slope.csv"
laterals_table_path = rpath + "/inputs/1d/topo/no_laterals.txt"
laterals_path = rpath + "/inputs/1d/hydro/"
boundary_condition_table = rpath + "/inputs/1d/topo/boundaryconditions.txt"
boundary_condition_path = rpath + "/inputs/1d/hydro/"

## result files
onedresultpath = rpath + "/results/1d/"
usbcpath = rpath + "/results/USbnd/"
# one_min_result_path = rpath + "/results/"
customized_runs_path = rpath + "/results/customized_results/"
rrmpath = rpath + "/inputs/rrm/hm_location"
twodresultpath = rpath + "/results/2d/zip/"

## gauges files
gauges_file = rpath + "/inputs/gauges/gauges.csv"
wl_files_rpath = rpath + "/inputs/gauges/water_level/"
q_files_rpath = rpath + "/inputs/gauges/discharge/"
# %% gauges
novalue = -9
start = "1955-01-01"
end = "1955-03-21"
Calib = RC.Calibration("HM", version=3)
Calib.readGaugesTable(gauges_file)
# read the gauges data
Calib.readObservedQ(
    q_files_rpath,
    start,
    end,
    novalue,
    file_extension=".txt",
    gauge_date_format="'%Y-%m-%d'",
)
Calib.readObservedWL(
    wl_files_rpath,
    start,
    end,
    novalue,
    file_extension=".txt",
    gauge_date_format="'%Y-%m-%d'",
)
# sort the gauges table based on the segment
Calib.hm_gauges.sort_values(by="id", inplace=True, ignore_index=True)
# %% create the river object
start = "1955-1-1"
rrmstart = "1955-1-1"
River = R.River("HM", version=3, start=start, rrm_start=rrmstart)
# %%
path = r"C:\gdrive\01Algorithms\Hydrology\Hapi\examples\Hydrodynamic-models\test_case\processing\def1D-1segment.yaml"
River.read_config(path)
# %%
# read the data of the river
"""the hourly results"""
River.one_d_result_path = onedresultpath
River.us_bc_path = usbcpath
"""the 1min results if exist"""
# River.one_min_result_path = one_min_result_path
"""river slope, cross-sections, and river network"""
River.read_slope(river_slope)
River.read_xs(xs_file)
River.read_river_network(river_network)
"""If you run part of the river and want to use its results as a boundary conditions for another run """
River.customized_runs_path = customized_runs_path
""" the results of the rain-runoff model"""
River.rrm_path = rrmpath
"""2D model results"""
# River.two_d_result_path = two_d_result_path
# River.compressed = True
# %% Interface
# The interface between the rainfall-runoff model and the hydraulic model
IF = Interface("Rhine", start=start)
IF.read_xs(xs_file)
IF.read_river_network(river_network)
IF.read_laterals_table(laterals_table_path)
IF.read_laterals(laterals_path, date_format="%d_%m_%Y")
IF.read_boundary_conditions_table(boundary_condition_table)
IF.read_boundary_conditions(path=boundary_condition_path, date_format="%d_%m_%Y")
# %% river segment
""" Write the segment-ID you want to visualize its results """
SubID = 1
Sub = R.Reach(SubID, River)
Sub.get_flow(IF)

## read RIM results
"""
read the 1D result file and extract only the first and last xs wl and hydrograph

if the results exists in a separate path than the project path (not in the results/1d) provide the new results path here
"""
# path = "F:/RFM/ClimXtreme/rim_base_data/setup/rhine/results/1d/New folder/"
Sub.read_1d_results()
# %% Select the gauge
"""
if the river segment has more than one gauge change this variable to the gauge
you want
"""
gaugei = 0

try:
    # get the gauges that are in the segment
    gauges = Calib.hm_gauges.loc[Calib.hm_gauges["id"] == SubID, :]
    gauges.index = range(len(gauges))
    stationname = gauges.loc[gaugei, "oid"]
    gaugename = str(gauges.loc[gaugei, "name"])
    gaugexs = gauges.loc[gaugei, "xsid"]
    segment_xs = str(SubID) + "_" + str(gaugexs)
    Laterals = Sub.get_laterals(gaugexs)
    print(print(gauges))
except KeyError:
    print("No gauge - choose another gauge to compare")
    gaugexs = Sub.last_xs
    segment_xs = str(SubID) + "_" + str(gaugexs)
    # get the gauge of the upstream segment
    """ write the segment id you want to get its data"""
    OtherSubID = 17
    gauges = Calib.hm_gauges.loc[Calib.hm_gauges["id"] == OtherSubID, :]
    gauges.index = range(len(gauges))
    stationname = gauges.loc[gaugei, "oid"]
    gaugename = str(gauges.loc[gaugei, "name"])
# %% Extract Results at the gauges
"""
the purpose of this part is read the 1D result files, Extract the cross
sections names, Number of cross sections, number of simulated days, detect if
there are Missing days (days which discharge was less than the algorithm
threshold so calculations were not made to save time) therefore these days has
to be filled with zero values
"""
# read rainfall runoff model result
# check if there is a rainfall runoff hydrograph with the name of the segment
try:
    Sub.read_rrm_hydrograph(
        stationname,
        date_format="'%Y-%m-%d'",
        location=2,
        path2=rpath + "/inputs/rrm/rrm_location",
    )
except:
    print("there is no Rainfall runoff hydrograph for this segment")

try:
    # read the 1D result file and extract only the first and last xs wl
    # and hydrograph
    # Path = "F:/RFM/mHM2RIM_testcase/RIM/results/1d/finished/"
    Sub.read_1d_results(xsid=gaugexs)  # ,Path = Path,FromDay = 18264, ToDay=18556
    print("Extract the XS results")
except:
    # read results of at the gauge
    CalibPath = "F:/RFM/mHM2RIM_testcase/RIM/results/calibration/"
    Calib.readCalirationResult(segment_xs, CalibPath)
    print("calibration result of the XS is read")

# read US boundary  hydrographs
Sub.read_us_hydrograph()
# Sum the laterals and the BC/US hydrograph
Sub.get_total_flow(gaugexs)
# %% Plot the Discharge hydrograph
hmorder = 11
gaugeorder = 7
rrmorder = 8
latorder = 9
ushorder = 10
xsorder = 9

# Specific XS
specificxs = False

start = str(Sub.first_day)[:-9]
end = str(Sub.last_day)[:-9]

fig, ax = Sub.plot_q(
    Calib,
    gaugexs,
    start,
    end,
    stationname,
    gaugename,
    segment_xs,
    plotgauge=True,
    gaugeorder=gaugeorder,
    gaugestyle=12,
    plotlaterals=True,
    latorder=latorder,
    ushcolor="#DC143C",
    plotus=True,
    ushorder=ushorder,
    ushstyle=11,
    specificxs=specificxs,
    xsorder=xsorder,
    plotrrm=True,
    rrm2color="orange",
    rrmorder=rrmorder,
    linewidth=5,
    figsize=(7, 6),
    hmorder=hmorder,
    xlabels=5,
)

# performance criteria
Filter = False
startError = start
endError = end
# startgauge = gauges.loc[gaugei, 'Qstart']
# endgauge = gauges.loc[gaugei, 'Qend']

Sub.calculate_q_metrics(
    Calib, stationname, gaugexs, filter=Filter, start=startError, end=endError
)

# plt.savefig(saveto + "/Segment-" + str(Reach.id) + "-" +
#             str(gauges.loc[gaugei, 'name']) + "-Q-C-" +
# str(dt.datetime.now())[0:11] + ".png")
# %% Hydrograph progression in a segment
xss = []
start = str(Sub.first_day)[:-9]
end = "1955-03-01"
fromxs = None
toxs = None
fig, ax = Sub.plot_hydrograph_progression(
    xss,
    start,
    end,
    from_xs=fromxs,
    to_xs=toxs,
    line_width=2,
    spacing=20,
    fig_size=(6, 4),
    x_labels=5,
)

# plt.savefig(saveto + "/Progression-" + str(Reach.id) + "-" +
#             str(gauges.loc[gaugei, 'name']) +
#             str(dt.datetime.now())[0:11] + ".png")
# %% Water Level
start = str(Sub.first_day.date())
end = str(Sub.last_day.date())

Sub.plot_wl(Calib, start, end, gaugexs, stationname, gaugename, plotgauge=True)

startError = start
endError = end

Sub.calculate_wl_metrics(
    Calib,
    stationname,
    gaugexs,
    Filter=False,
    start=startError,
    end=endError,
)

# plt.savefig(saveto + "/Segment-" + str(Reach.id) + "-"
#             + str(gauges.loc[gaugei,'name']) +
#             "-WL-C-" + str(dt.datetime.now())[0:11] + ".png")
# %% calibration (the bed level change the levels)
# NoSegments = 1
# read theoriginal slope and XS files
Calib.cross_sections = River.cross_sections
Calib.slope = River.slope

BedlevelDS = 88
Manning = 0.06
BC_slope = -0.03
Calib.calculateProfile(SubID, BedlevelDS, Manning, BC_slope)
# River.cross_sections.to_csv(tolopogy_file + "/xs_rhine2.csv", index=False, float_format="%.3f")
# River.slope.to_csv(tolopogy_file + "/slope2.csv",header=None,index=False)
# %% Smooth cross section
Calib.cross_sections = River.cross_sections[:]
Calib.smoothMaxSlope(SubID)
Calib.smoothBedLevel(SubID)
Calib.downWardBedLevel(SubID, 0.05)
# Calib.SmoothBankLevel(SubID)
# Calib.SmoothFloodplainHeight(SubID)
Calib.smoothBedWidth(SubID)
# Calib.CheckFloodplain()
# Calib.cross_sections.to_csv(tolopogy_file + "/XS2.csv", index=None, float_format="%.3f")
# %% customized Run result saveing
# the last cross section results to use it in calibration
"""
this part is to save the results of the last cross section of this sub-basin
to use it as a cusomized results later to run the down stream sub-basins
without the need to run all the upstream sub-basins

you have to un comment the following two lines
"""
# Path = wpath + "/results/customized_results/"
Sub.save_hydrograph(Sub.last_xs)  # Path
# %% Filters
"""
check the max sf
"""
## calculate the water surface difference
# wl = Reach.results_1d.loc[Reach.results_1d.index[i],'wl']
sf = [
    (
        Sub.results_1d.loc[Sub.results_1d.index[i], "wl"]
        - Sub.results_1d.loc[Sub.results_1d.index[i + 1], "wl"]
    )
    / 500
    for i in range(len(Sub.results_1d.index) - 1)
]
sf = sf + [np.mean(sf)]
Sub.results_1d["sf"] = sf

print(Sub.results_1d[Sub.results_1d["sf"] == Sub.results_1d["sf"].max()])
print(Sub.results_1d[Sub.results_1d["sf"] == Sub.results_1d["sf"].min()])

"""some filter to get where the min depth (dryness limit)"""

# dataX = Reach.results_1d[Reach.results_1d['xs'] == 700]
dataX = Sub.results_1d[Sub.results_1d["h"] == 0.01]
# dataX = Reach.results_1d[Reach.results_1d['xs'] == 121]
# %% get the boundary conditions
start = "1955-01-01"
end = "1955-03-21"

Sub.read_boundary_conditions(start=start, end=end)
# %% Visualize
fromxs = ""  # 16030
toxs = ""  # 16067

Vis = V(resolution="Hourly")

Vis.plotGroundSurface(
    Sub,
    floodplain=True,
    plot_lateral=True,
    xlabels_number=20,
    from_xs=fromxs,
    to_xs=toxs,
    option=2,
)
# %% cross-sections
fig, ax = Vis.plotCrossSections(
    Sub,
    bedlevel=True,
    from_xs=fromxs,
    to_xs=toxs,
    same_scale=True,
    text_spacing=[(1, 1), (1, 4)],
    plotting_option=3,
)
# %% Animation
""" periods of water level exceeds the bankfull_depth depth"""

start = "1955-02-10"
end = "1955-02-11"
from matplotlib import rc

Anim = Vis.WaterSurfaceProfile(
    Sub,
    start,
    end,
    fps=2,
    xlabels_number=5,
    from_xs=fromxs,
    to_xs=toxs,
    x_axis_label_size=10,
    text_location=(-1, -2),
    repeat=True,
)
plt.close()
# rc('animation', html='jshtml')
# rc
# %%
# TODO : create a test for SaveProfileAnimation function
ffmpegPath = "F:/Users/mofarrag/.matplotlib/ffmpeg-4.4-full_build/bin/ffmpeg.exe"
SavePath = saveto + "/" + str(Sub.id) + "-" + str(dt.datetime.now())[:13] + ".gif"
Vis.SaveProfileAnimation(Anim, Path=SavePath, fps=30, ffmpegPath=ffmpegPath)
# %% Read and plot the 1 min data

start = "1955-01-01"
end = "1955-01-10"

Sub.read_sub_daily_results(start, end, last_river_reach=True)
# %%
# negative values
# TODO : check CheckNegativeQ makes problem
# Reach.CheckNegativeQ(temporal_resolution = '1min')
# %% Plotting
start = "1955-01-01"
end = "1955-01-10"

fromxs = ""
toxs = ""

Vis = V(resolution="Hourly")
Anim = Vis.WaterSurfaceProfile1Min(
    Sub,
    start,
    end,
    interval=0.000000000000000000000000000000000001,
    from_xs=fromxs,
    to_xs=toxs,
)
# %% Q for all XS
"""
this part will plot the 1 min Q and H at all cross section of the given
sub-basin at a certain time to visualise the spikes where the algorithm switches bertween
calculating discharge with the calculated sf or using the min Sf

"""
date = "1955-01-05"
Vis.Plot1minProfile(Sub, date, nxlabels=20)
# %%  plot BC
date = "1955-01-05"
Sub.plot_bc(date)
# %% new table
"""
this part is to plot the geometric properties of the cross sectin
as the area and perimeter changes for each water depth whether it is
greater, equal or less than the bankfull depth and change in the calculated
discharge is the hydraulic radius is calculated for the whole cross section
or calculated for each one separately
first you have to print/ save the result of the rating curve from the RIM
algorithm
"""
# i = 19
# i = no
# ids = list(set(XS['swmmid']))
Res1 = "F:/mofarrag/Documents/01Model/01Files/Rhine/02_Models/RIM_Rhine/runcode/BC/"
table_new = pd.read_csv(Res1 + str(SubID) + "-BC.txt", header=None, delimiter=r"\s+")
# table_new = pd.read_csv(Res1+str(SubID)+"-BC.txt",header =None,delimiter = r'\s+')
# table_new = pd.read_csv(Res1+"343-BC00000000000.txt",header =None,delimiter = r'\s+')
table_new.columns = [
    "depth",
    "area_T",
    "perimeter_T",
    "area_U",
    "perimeter_U",
    "area_L",
    "perimeter_L",
    "Q_U",
    "Q_L",
    "Q_T",
]
# table_new['R'] = table_new ['area']/table_new ['perimeter']
# table_new['Q'] = (table['A*R^(2/3)']*((0.1/500)**0.5))/0.03
# table_new['v'] = table_new ['Q'] / table_new ['area']
# table_new['logQ'] = np.log10(table_new ['Q'])
# table_new['logH'] = np.log10(table_new ['depth'])

dbf = Sub.cross_sections["dbf"][Sub.cross_sections["xsid"] == Sub.xs_names[0]].values[0]
b = Sub.cross_sections["b"][Sub.cross_sections["xsid"] == Sub.xs_names[0]].values[0]
Abf = dbf * b
Pbf = b + 2 * dbf
# Qdbf = (1.0/0.03)*(Abf *((Abf/Pbf)**(2.0/3.0)))*((0.1/500)**0.5)


plt.figure(50, figsize=(15, 8))
# plt.plot(table_new['area_T'],table_new['depth'], label = 'Area_T', line_width = 5)
# plt.plot(table_new['area_U'],table_new['depth'], label = 'Area_U', line_width = 5)
# plt.plot(table_new['area_L'],table_new['depth'], label = 'Area_L', line_width = 5)

plt.plot(table_new["perimeter_T"], table_new["depth"], label="Perimeter_T", linewidth=5)
plt.plot(table_new["perimeter_U"], table_new["depth"], label="Perimeter_U", linewidth=5)
plt.plot(table_new["perimeter_L"], table_new["depth"], label="Perimeter_L", linewidth=5)


# plt.plot(table_new['Q_U'],table_new['depth'], label = 'Q_U', line_width = 5)
# plt.plot(table_new['Q_L'],table_new['depth'], label = 'Q_L', line_width = 5)
# plt.plot(table_new['Q_T'],table_new['depth'], label = 'Q_T', line_width = 5)


# plt.plot(table['logQ'],table['logH'], label = 'Area', line_width = 5)

plt.ylabel("Depth (m)", fontsize=20)
plt.ylim([0, 8])
plt.xlim([0, table_new["Q_T"].loc[table_new["depth"] == 8].values[0] + 5])
plt.hlines(
    Sub.cross_sections["dbf"]
    .loc[Sub.cross_sections["xsid"] == Sub.xs_names[0]]
    .values[0],
    0,
    table_new["area_T"].loc[table_new["depth"] == 5].values[0],
    linewidth=5,
)
plt.annotate(
    "Dbf = "
    + str(
        Sub.cross_sections["dbf"]
        .loc[Sub.cross_sections["xsid"] == Sub.xs_names[0]]
        .values[0]
    ),
    xy=(
        table_new["perimeter_T"].loc[table_new["depth"] == 5].values[0] - 80,
        Sub.cross_sections["dbf"]
        .loc[Sub.cross_sections["xsid"] == Sub.xs_names[0]]
        .values[0]
        + 0.2,
    ),
    fontsize=20,
)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("Reach-basin" + str(SubID), fontsize=20)
plt.legend(fontsize=20)

# %% XS properties function results
Res = ""
table = pd.read_csv(
    Res + "table/" + str(SubID) + "-table.txt", header=None, delimiter=r"\s+"
)
table.columns = ["depth", "area", "perimeter", "A*R^(2/3)"]
table["R"] = table["area"] / table["perimeter"]
table["Q"] = (table["A*R^(2/3)"] * ((0.1 / 500) ** 0.5)) / 0.03
table["v"] = table["Q"] / table["area"]
table["logQ"] = np.log10(table["Q"])
table["logH"] = np.log10(table["depth"])

dbf = Sub.cross_sections["dbf"][Sub.cross_sections["xsid"] == Sub.xs_names[0]].values[0]
b = Sub.cross_sections["b"][Sub.cross_sections["xsid"] == Sub.xs_names[0]].values[0]
Abf = dbf * b
Pbf = b + 2 * dbf
Qdbf = (1.0 / 0.03) * (Abf * ((Abf / Pbf) ** (2.0 / 3.0))) * ((0.1 / 500) ** 0.5)


table_zone2 = table[table["depth"] > dbf]
table_zone3 = table_zone2["Q"].min()
table["order"] = list(range(1, len(table) + 1))
dbfloc = list(
    np.where(
        table["depth"]
        <= Sub.cross_sections["dbf"][
            Sub.cross_sections["xsid"] == Sub.xs_names[0]
        ].values[0]
    )
)[-1][-1]

# %% plotting
plt.figure(80, figsize=(15, 8))
plt.plot(table["area"], table["depth"], label="Area", linewidth=5)
plt.plot(table["perimeter"], table["depth"], label="Perimeter", linewidth=5)
plt.plot(table["R"], table["depth"], label="R", linewidth=5)
plt.plot(table["A*R^(2/3)"], table["depth"], label="A*(R^2/3)", linewidth=5)
plt.plot(table["Q"], table["depth"], label="Q", linewidth=5)
plt.plot(table["v"], table["depth"], label="velocity", linewidth=5)

# plt.plot(table['logQ'],table['logH'], label = 'Area', line_width = 5)

plt.ylabel("Depth (m)", fontsize=20)
plt.ylim([0, 5])
plt.xlim([0, table["perimeter"].loc[table["depth"] == 5].values[0] + 5])
plt.hlines(
    Sub.cross_sections["dbf"]
    .loc[Sub.cross_sections["xsid"] == Sub.xs_names[0]]
    .values[0],
    0,
    table["area"].loc[table["depth"] == 5].values[0],
    linewidth=5,
)
plt.annotate(
    "Dbf = "
    + str(
        Sub.cross_sections["dbf"]
        .loc[Sub.cross_sections["xsid"] == Sub.xs_names[0]]
        .values[0]
    ),
    xy=(
        table["perimeter"].loc[table["depth"] == 5].values[0] - 80,
        Sub.cross_sections["dbf"]
        .loc[Sub.cross_sections["xsid"] == Sub.xs_names[0]]
        .values[0]
        + 0.2,
    ),
    fontsize=20,
)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend()
# %% Q2H
# q = BC_q.loc[dt.datetime(counter[i].year,counter[i].month,counter[i].day)].loc[475]
# np.where(table['Q'] <= q)
