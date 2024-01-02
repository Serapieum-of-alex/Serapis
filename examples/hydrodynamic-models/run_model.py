"""Created on Tue Mar 16 22:27:13 2021.

@author: mofarrag
"""
import datetime as dt

# from IPython import get_ipython
# get_ipython().magic('reset -f')
import os

import numpy as np
import pandas as pd

import Hapi.hm.river as R
from Hapi.hm.interface import Interface
from Hapi.hm.saintvenant import GVF

Comp = "F:/RFM/mHM2RIM_testcase"
os.chdir(Comp + "/RIM/processing")
# %% Paths
# the working directory of the project
wpath = Comp + "/RIM"
RRMPath = Comp + "/base_data/calibration/mHM"
RIM2Files = wpath + "/inputs/1d/topo/"
BaseDataPath = Comp + "/base_data"
savepath = wpath + "/results/customized_results/"

start = "1952-1-1"
RRMstart = "1952-1-1"

River = R.River("RIM", Version=3, start=start, RRMstart=RRMstart, RRMdays=23192)

River.OneDResultPath = (
    Comp + "/base_data/calibration/calibration_results/all_results/20210315/"
)
River.USbndPath = wpath + "/results/USbnd/"
River.OneMinResultPath = wpath + "/results/"
River.TwoDResultPath = wpath + "/results/2d/zip/"
River.CustomizedRunsPath = wpath + "/results/customized_results/"
River.Compressed = True
River.RRMPath = RRMPath

Path = wpath + "/processing/def1D.txt"
River.read_1d_config_file(Path)
# %%
IF = Interface("Rhine")
IF.read_xs(RIM2Files + "/XS.csv")
IF.read_river_network(RIM2Files + "/rivernetwork.txt")
IF.read_laterals_table(wpath + "/inputs/1d/topo/laterals-segment24.txt")
IF.read_laterals(Path=wpath + "/inputs/1d/hydro/", date_format="%d_%m_%Y")
IF.read_boundary_conditions_table(
    wpath + "/inputs/1d/topo/BonundaryConditions-segment24.txt"
)
IF.read_boundary_conditions(Path=wpath + "/inputs/1d/hydro/", date_format="%d_%m_%Y")
# %% Reach-basin
""" Write the Reach-ID you want to visualize its results """

River.RoutedQ = np.zeros(shape=(River.NoTimeSteps, River.NoSeg))
# sum of all US routedQ
River.DirectUS = np.zeros(shape=(River.NoTimeSteps, River.NoSeg))
# sum of the DirectUS and BC
River.TotalUS = np.zeros(shape=(River.NoTimeSteps, River.NoSeg))

ICflag = np.zeros(shape=(River.NoSeg))
storewl = np.zeros(shape=(River.XSno, 2))

# HydrologicalTempRes = 24
# OneMinTimeSteps = 60 * HydrologicalTempRes

for i in range(River.NoSeg):
    SubID = River.Segments[i]
    Sub = R.Reach(SubID, River, RunModel=True)
    Sub.get_flow(IF, SubID)
    # HQ : is a rating curve table contains discharge at the first column and coresponding
    # water depth at the second column
    # HQ is allocated inside the RatingCurve subroutine
    Sub.get_rating_curve()
    # get the area and perimeters of the cross section if the water level is at
    # min of the intermediate point left and right and then the max of both
    Sub.get_xs_geometry()
    Sub.get_us_hydrograph(River)

    s = River.date_to_ordinal(str(River.SimStart)[:-9])

    e = River.date_to_ordinal(str(River.SimEnd + dt.timedelta(days=1))[:-9])
    for step in range(s, e):
        step = River.ordinal_to_date(step)

        print("Step-" + str(step))
        step_ind = River.date_to_ordinal(step)
        # get the max discharge
        Q = Sub.Laterals.loc[step : step + dt.timedelta(days=1), :]
        # index starts from 1
        Q.loc[:, "US"] = Sub.us_hydrographs[step_ind - 1 : step_ind + 1]

        if Q.sum(axis=1).values.max() > River.D1["MinQ"]:
            # interpolate to 1 min resolution
            ind = pd.date_range(step, step + dt.timedelta(days=1), freq="1Min")
            MinQ = pd.DataFrame(index=ind, columns=Q.columns)
            MinQ.loc[:, :] = Q.loc[:, :].resample("1Min").mean().interpolate("linear")
            # convert the UShydrograph into water depth
            Hbnd = Sub.h_to_q(MinQ["US"].values)

            if step_ind == 1 or ICflag[i] == 0:
                inih = 0.01
            else:
                # initial water depth
                inih = inih

            ICflag[i] = 1

            q, h, wl = GVF(Sub, River, Hbnd, 60, 500, inih, storewl, MinQ)

            River.RoutedQ[step_ind - 1, River.Segments.index(SubID)] = q[
                Sub.XSno - 1, 0
            ]
            River.RoutedQ[step_ind, River.Segments.index(SubID)] = q[
                Sub.XSno - 1, River.TS - 1
            ]

            inih = h[:, River.TS - 1]
