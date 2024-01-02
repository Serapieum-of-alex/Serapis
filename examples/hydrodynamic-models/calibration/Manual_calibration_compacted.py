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
from Hapi.visualizer import Visualize as V

"""change directory to the processing folder inside the project folder"""
os.chdir(r"F:\01Algorithms\Hydrology\HAPI")
rpath = os.path.abspath(os.getcwd() + "/examples/Hydrodynamic models/test_case")
saveto = rpath
# %% gauges
GaugesF = rpath + "/inputs/gauges/gauges.csv"
WLGaugesPath = rpath + "/inputs/gauges/water_level/"
QgaugesPath = rpath + "/inputs/gauges/discharge/"

novalue = -9
start = "1955-01-01"
end = "1955-03-21"
Calib = RC.Calibration("HM", version=3)
Calib.read_gauges_table(GaugesF)
Calib.read_observed_q(
    QgaugesPath,
    start,
    end,
    novalue,
    file_extension=".csv",
    gauge_date_format="'%Y-%m-%d'",
)
Calib.read_observed_wl(
    WLGaugesPath,
    start,
    end,
    novalue,
    file_extension=".csv",
    gauge_date_format="'%Y-%m-%d'",
)
# sort the gauges table based on the segment
Calib.hm_gauges.sort_values(by="id", inplace=True, ignore_index=True)
# %% Paths
# the working directory of the project
RIM2Files = rpath + "/inputs/1d/topo/"
savepath = rpath + "/results/customized_results/"

start = "1955-1-1"
rrmstart = "1955-1-1"

River = R.River("HM", version=3, start=start, rrmstart=rrmstart)
# read the configuration file
River.read_1d_config_file(rpath + "/processing/def1D-1segment_very_steep.txt")
# %% Interface
IF = Interface("Rhine", start=start)
IF.read_xs(RIM2Files + "/xs_same_downward-3segment.csv")
IF.read_river_network(RIM2Files + "/rivernetwork-3segments.txt")
IF.read_laterals_table(rpath + "/inputs/1d/topo/laterals.txt")
IF.read_laterals(path=rpath + "/inputs/1d/hydro/", date_format="%d_%m_%Y")
IF.read_boundary_conditions_table(rpath + "/inputs/1d/topo/boundaryconditions.txt")
IF.read_boundary_conditions(path=rpath + "/inputs/1d/hydro/", date_format="%d_%m_%Y")
# %% Reach-basin
""" Write the Reach-ID you want to visualize its results """
SubID = 1
Sub = R.Reach(SubID, River)
Sub.get_flow(IF)
# %% read RIM results
"""
read the 1D result file and extract only the first and last xs wl and
hydrograph
"""
path = r"F:\02Case-studies\ClimXtreme\rim_base_data\src\rim\test_case\ideal_case/results/1d/"
Sub.read_1d_results(path=path)  # path=path,XSID=gaugexs
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
    Calib.read_caliration_result(segment_xs, CalibPath)
    print("calibration result of the XS is read")

# read US boundary  hydrographs
Sub.read_us_hydrograph()
# Sum the laterals and the BC/US hydrograph
Sub.get_total_flow(gaugexs)
# %% Discharge
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
fromxs = ""
toxs = ""
fig, ax = Sub.plot_hydrograph_progression(
    xss,
    start,
    end,
    fromxs=fromxs,
    toxs=toxs,
    linewidth=2,
    spacing=20,
    figsize=(6, 4),
    x_labels=5,
)

# plt.savefig(saveto + "/Progression-" + str(Reach.id) + "-" +
#             str(gauges.loc[gaugei, 'name']) +
#             str(dt.datetime.now())[0:11] + ".png")
# %% Water Level
start = str(Sub.first_day)[:-9]
end = str(Sub.last_day)[:-9]

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
Calib.calculate_profile(SubID, BedlevelDS, Manning, BC_slope)
# River.cross_sections.to_csv(RIM2Files + "/xs_rhine2.csv", index=False, float_format="%.3f")
# River.slope.to_csv(RIM2Files + "/slope2.csv",header=None,index=False)
# %% Smooth cross section
Calib.cross_sections = River.cross_sections[:]
Calib.smooth_max_slope(SubID)
Calib.smooth_bed_level(SubID)
Calib.downward_bed_level(SubID, 0.05)
# Calib.SmoothBankLevel(SubID)
# Calib.SmoothFloodplainHeight(SubID)
Calib.smooth_bed_width(SubID)
# Calib.CheckFloodplain()
# Calib.cross_sections.to_csv(RIM2Files + "/XS2.csv", index=None, float_format="%.3f")
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
