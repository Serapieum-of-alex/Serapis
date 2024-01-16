"""River Module.

river module to read the river data and do hydraulic analysis
"""
import datetime as dt
import os
from bisect import bisect
from pathlib import Path
from typing import Any, Optional, Tuple, Union, List, Callable
import yaml
from loguru import logger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from pandas.core.frame import DataFrame
from serapeum_utils.utils import class_attr_initialize, class_method_parse
from statista import metrics as metrics
from statista.distributions import Distributions
from serapis.serapis_warnings import SilencePandasWarning
from serapis.saintvenant import SaintVenant
from cleopatra.styles import Styles
from serapis.plot.visualizer import Visualize as V

SilencePandasWarning()

hours = list(range(1, 25))


class River:
    """River.

    River class reads all the data of the river (cross-sections,
    simulation results) and analyze the results and do visualization
    """

    initial_args = dict(
        name={"type": str},
        version={"default": 3, "type": int},
        dto={"default": 60, "type": int},
        dx={"default": 500, "type": int},
        start={"default": "1950-1-1", "type": str},
        days={"default": 36890, "type": int},  # 100 years
        rrm_start={"default": None, "type": str},
        rrm_days={"default": 36890, "type": int},  # 100 years
        left_overtopping_suffix={"default": "_left.txt", "type": str},
        right_overtopping_suffix={"default": "_right.txt", "type": str},
        depth_prefix={"default": "DepthMax", "type": str},
        duration_prefix={"default": "Duration", "type": str},
        return_period_prefix={"default": "ReturnPeriod", "type": str},
        compressed={"default": True, "type": bool},
        fmt={"default": "%Y-%m-%d", "type": str},
        one_d_result_path={"default": "/results/1d", "type": str},
        two_d_result_path={"default": "/results/2d", "type": str},
    )

    river_attributes = dict(
        one_min_result_path=None,
        us_bc_path=None,
        first_day=None,
        referenceindex_results=None,
        wd=None,
        XSF=None,
        laterals_file=None,
        BCF=None,
        river_network_file=None,
        slope_file=None,
        NoSeg=None,
        CalibrationF=None,
        Coupling1D2DF=None,
        RunMode=None,
        Subid=None,
        Customized_BC_F=None,
        ResultsDetails=None,
        RRMTemporalResolution=None,
        HMTemporalResolution=None,
        HMStoreTimeStep=None,
        TS=None,
        SimStartIndex=None,
        SimEndIndex=None,
        SimStart=None,
        SimEnd=None,
        OneDTempR=None,
        D1=None,
        D2=None,
        cross_sections=None,
        xsno=None,
        xs_names=None,
        q_bc_1min=None,
        h_bc_1min=None,
        h=None,
        q=None,
        from_beginning=None,
        first_day_results=None,
        last_day=None,
        days_list=None,
        id=None,
        QBC=None,
        HBC=None,
        # usbc=None,
        # dsbc=None,
        results_1d=None,
        Q=None,
        H=None,
        slope=None,
        event_index=None,
        river_network=None,
        SP=None,
        customized_runs_path=None,
        Segments=None,
        RP=None,
        rrm_path=None,
        reach_ids=None,
        customized_runs_config=None,
        parameters=None,
        results_config=None,
        rrm_paths=None,
        rrm_config=None,
        river_1d_paths=None,
        river_1d_config=None,
        config=None,
        results_paths=None,
        one_min_results_config=None,
        hourlt_results_config=None,
    )

    @class_method_parse(initial_args)
    @class_attr_initialize(river_attributes)
    def __init__(
        self,
        name: str,
        version: int = 3,
        start: str = "1950-1-1",
        end: Union[int, str] = None,
        *args,
        **kwargs,
    ):
        """River.

            to instantiate the river class, you need to provide the following parameters

        Parameters
        ----------
        name: [str]
            name of the river.
        version: [integer], optional
            version of the hydraulic model. The default is 3.
        start: [str], optional
            start date. The default is "1950-1-1".
        days: [integer], optional
            length of the simulation in days. The default is 36890.
        rrm_start: [str], optional
            the start date of the rainfall-runoff data. The default is
            "1950-1-1".
        rrm_days: [integer], optional
            the length of the data of the rainfall-runoff data in days.
            The default is 36890.
        dto : [integer]
            time step (sec) of the 1d routing model. default is 60 seconds.
        left_overtopping_suffix: [str], optional
            the prefix you used to name the overtopping form the left bank
            files.
            The default is "_left.txt".
        right_overtopping_suffix: TYPE, optional
            the prefix you used to name the overtopping form the right bank
            files. The default is "_right.txt".
        depth_prefix: [str], optional
            the prefix you used to name the Max depth raster result maps.
            The default is "DepthMax".
        duration_prefix: [str], optional
            the prefix you used to name the inundation duration raster result
            maps. The default is "Duration".
        return_period_prefix: [str], optional
            the prefix you used to name the Return Period raster result maps.
            The default is "ReturnPeriod".
        compressed: [bool], optional
            True if the result raster/ascii files are compressed. The default
            is True.
        one_d_result_path: [str], optional
            path to the folder where the 1D river routing results exist.
            The default is ''.
        two_d_result_path: [str], optional
            path to the folder where the 1D river routing results exist.
            The default is ''.
        fmt: [string]
            format of the date. fmt="%Y-%m-%d %H:%M:%S"
        """
        # positional arguments
        assert isinstance(start, str), "start argument has to be string"
        assert isinstance(version, int), "version argument has to be integer number"

        self.name = name
        self.version = version
        self.start = dt.datetime.strptime(start, self.fmt)

        if end is None:
            self.end = self.start + dt.timedelta(days=self.days)
        else:
            self.end = dt.datetime.strptime(end, self.fmt)
            self.days = (self.end - self.start).days

        self.dt = self.dto
        if self.dt < 60:
            self.freq = str(self.dt) + "S"
        else:
            self.freq = str(int(self.dt / 60)) + "Min"
        # ----------------------------------------------------
        ref_ind = pd.date_range(self.start, self.end, freq="D")
        # the last day is not in the result day Ref_ind[-1]
        # write the number of days + 1 as python does not include the last
        # number in the range
        # 19723 days so write 19724
        if self.days == 1:
            self.days = 2
            self.reference_index = pd.DataFrame(index=list(range(1, self.days + 1)))
            self.reference_index["date"] = ref_ind
        else:
            self.reference_index = pd.DataFrame(index=list(range(1, self.days + 1)))
            self.reference_index["date"] = ref_ind[:-1]

        if self.rrm_start is None:
            self.rrm_start = self.start
        else:
            self.rrm_start = dt.datetime.strptime(self.rrm_start, self.fmt)

        self.rrm_end = self.rrm_start + dt.timedelta(days=self.rrm_days)
        ref_ind = pd.date_range(self.rrm_start, self.rrm_end, freq="D")
        self.rrm_reference_index = pd.DataFrame(index=list(range(1, self.rrm_days + 1)))
        self.rrm_reference_index["date"] = ref_ind[:-1]
        self.no_time_steps = len(self.rrm_reference_index)

        self.indsub = pd.date_range(self.start, self.end, freq=self.freq)

    def ordinal_to_date(self, index: int):
        """IndexToDate.

        IndexToDate takes an integer number and returns the date corresponding
        to this date based on a time series starting from the "start" attribute
        of the River object and for a length of the value of the "days" attribute

        Parameters
        ----------
        index : [Integer]
            Integer number ranges from 1 and max value of the value of the
            attribute "days" of the River object.

        Returns
        -------
        [Date time ]
            date object.
        """
        # convert the index into date
        return self.reference_index.loc[index, "date"]

    @property
    def cross_sections(self):
        """River cross-sections."""
        return self._cross_sections

    @cross_sections.setter
    def cross_sections(self, value):
        self._cross_sections = value

    @property
    def event_index(self):
        """Event index."""
        return self._event_index

    @event_index.setter
    def event_index(self, value: DataFrame):
        """Event index."""
        self._event_index = value

    @property
    def usbc(self):
        """Upstream Boundary condition."""
        return self._usbc

    # @usbc.setter
    # def usbc(self, value):
    #     self._usbc = value

    @property
    def dsbc(self):
        """Downstream boundary condition."""
        return self._dsbc

    def _get_date(self, day, hour):
        """Get date for the 1D result data frame.

        Parameters
        ----------
        day: [int]
            ordinal date
        hour: [int]
            hour.

        Returns
        -------
        datetime
        """
        return self.ordinal_to_date(day) + dt.timedelta(hours=hour - 1)

    def date_to_ordinal(self, date: Union[dt.datetime, str], fmt: str = "%Y-%m-%d"):
        """date_to_ordinal.

            date_to_ordinal takes a date and returns the order of the days in the
            time series. The time series starts from the value of the "start" for
            a length of "days" value

        Parameters
        ----------
        date: [string/date time object]
            string in the format of "%Y-%m-%d" or a date time object.
        fmt: [string]
            format of the date. fmt="%Y-%m-%d %H:%M:%S"
        Returns
        -------
        [Integer]
            the order of the date in the time series.
        """
        if isinstance(date, str):
            date = dt.datetime.strptime(date, fmt)
        try:
            return np.where(self.reference_index["date"] == date)[0][0] + 1
        except:
            raise ValueError(
                f"The input date {date} is out of the range"
                f"Simulation is between {self.reference_index.loc[1, 'date']} and "
                f"{self.reference_index.loc[len(self.reference_index), 'date']}"
            )

    def index_to_date_rrm(self, index: int):
        """index_to_date_rrm.

        IndexToDate takes an integer number and returns the date coresponding
        to this date based on a time series starting from the "start" attribute
        of the River object and for a length of the value of the "days" attribute

        Parameters
        ----------
        index : [Integer]
            Integer number ranges from 1 and max value of the value of the attribute
            "days" of the River object.

        Returns
        -------
        [Date time ]
            date object.
        """
        # convert the index into date
        return self.reference_index.loc[index, "date"]

    def date_to_index_rrm(self, date: Union[str, dt.datetime], fmt: str = "%Y-%m-%d"):
        """date_to_Index_rrm.

            date_to_Index_rrm takes a date and returns the order of the days in the
            time series. The time series starts from the value of the "start" for
            a length of "days" value

        Parameters
        ----------
        date: [string/date time object]
            string in the format of "%Y-%m-%d" or a date time object.
        fmt: [string]
            format of the date. fmt="%Y-%m-%d %H:%M:%S"
        Returns
        -------
        [Integer]
            the order of the date in the time series.
        """
        if isinstance(date, str):
            date = dt.datetime.strptime(date, fmt)
        return np.where(self.reference_index["date"] == date)[0][0] + 1

    @staticmethod
    def round(number, round_to):
        """Round fload number."""
        return round(number / round_to) * round_to

    def read_config(self, path):
        """Read the hydraulic model configuration file.

        Parameters
        ----------
        path: [str]
            path to the configuration file (yaml files)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"The Configuration file You have entered: {path} does not exist"
            )
        with open(path, "r") as stream:
            config = yaml.safe_load(stream)
        # project directory
        self.wd = config.get("project directory")
        self.config = config
        # river
        river_files = config.get("river description")
        self.river_1d_config = river_files
        river_rdir = Path(river_files.get("root directory"))
        self.river_1d_paths = dict(
            river_rdir=river_rdir,
            xs_file=river_rdir.joinpath(river_files.get("cross sections")),
            river_network=river_rdir.joinpath(river_files.get("river network")),
            river_slope=river_rdir.joinpath(river_files.get("slope")),
            coupling_2d=river_rdir.joinpath(river_files.get("1D-2D coupling")),
            calibration_table=river_rdir.joinpath(river_files.get("results locations")),
        )
        # rainfall runoff model
        rrm_files = config.get("rainfall-runoff files")
        self.rrm_config = rrm_files
        rrm_rdir = Path(rrm_files.get("root directory"))
        rrm_results = rrm_files.get("river routing blocked results")
        self.rrm_paths = dict(
            rrm_rdir=rrm_rdir,
            laterals_table_path=rrm_rdir.joinpath(rrm_files.get("laterals")),
            boundary_condition_table=rrm_rdir.joinpath(
                rrm_files.get("boundary condition")
            ),
            laterals_dir=rrm_rdir,
            boundary_condition_path=rrm_rdir,
            rrm_location_1=Path(rrm_results.get("location-1")),  # rrm_path
            rrm_location_2=Path(rrm_results.get("location-2")),
        )
        # result files
        results_files = config.get("Results 1D")
        self.results_config = results_files
        results_rdir = Path(results_files.get("root directory"))
        hourly_results = results_files.get("hourly")
        self.hourly_results_config = hourly_results
        one_min_results = results_files.get("one min")
        self.one_min_results_config = one_min_results
        # 2D
        results_files = config.get("Results 2D")
        self.results_paths = dict(
            results_rdir=results_rdir,
            one_d_result_path=results_rdir.joinpath(hourly_results.get("folder")),
            one_min_result_path=results_rdir,
            us_bc_path=results_rdir.joinpath(one_min_results.get("usbc").get("folder")),
            two_d_result_path=Path(results_files.get("root directory")),
        )
        # parameters
        parameters = config.get("simulation parameters")
        self.parameters = parameters
        customized_runs = parameters.get("customized simulation")
        self.customized_runs_config = customized_runs
        customized_runs_path = Path(customized_runs.get("previous run results"))
        self.customized_runs_path = customized_runs_path

    def read_1d_config_file(self, path: str):
        """read_1d_config_file.

        Read the configuration file

        Parameters
        ----------
        path : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        """
        file = open(path)
        content = file.readlines()
        file.close()
        # project path
        self.wd = content[1][:-1]
        # cross-sections file
        self.XSF = content[4][:-1]
        self.read_xs(f"{self.wd}/inputs/1d/topo/{self.XSF}")
        # Laterals file, BC file
        self.laterals_file, self.BCF = content[6][:-1].split(" ")
        # river_network file
        self.river_network_file = content[8][:-1]
        self.read_river_network(f"{self.wd}/inputs/1d/topo/{self.river_network_file}")
        # Slope File
        self.slope_file = content[10][:-1]
        self.read_slope(f"{self.wd}/inputs/1d/topo/{self.slope_file}")
        self.NoSeg = len(self.slope)
        # Calibration file
        self.calibration_file = content[12][:-1]
        # 1D-2D Coupling file
        self.Coupling1D2DF = content[14][:-1]
        # Run mode
        run_mode, sub_id = content[16][:-1].split("#")[0].strip().split(" ")
        self.RunMode = int(run_mode)
        # Segment id
        self.sub_id = int(sub_id)
        # Customized Run file
        self.Customized_BC_F = content[18][:-1]
        self.customized_runs_path = content[19][:-1]

        # Results
        delete_old = content[22][:-1]
        delete_old = int(delete_old)
        save_hourly, save_overtopping_tot, save_overtopping_detailed = content[24][
            :-1
        ].split(" ")
        save_hourly = int(save_hourly)
        save_overtopping_tot = int(save_overtopping_tot)
        save_overtopping_detailed = int(save_overtopping_detailed)
        save_q, save_wl, save_usbc = content[26].split("#")[0][:-1].split(" ")
        save_q = int(save_q)
        save_wl = int(save_wl)
        save_usbc = int(save_usbc)

        self.ResultsDetails = dict(
            DeleteOld=delete_old,
            SaveHourly=save_hourly,
            SaveOverToppingTot=save_overtopping_tot,
            SaveOverToppingDetailed=save_overtopping_detailed,
            SaveQ=save_q,
            SaveWL=save_wl,
            SaveUSBC=save_usbc,
        )

        # Temporal Resolution
        self.RRMTemporalResolution = content[31].split("#")[0][:-1]
        self.HMTemporalResolution = int(content[32].split("#")[0][:-1])
        self.HMStoreTimeStep = int(content[33].split("#")[0][:-1])

        if self.RRMTemporalResolution == "Daily":
            # number of time steps for the 1D model
            self.TS = 24 * self.HMTemporalResolution

        start, end = content[35][:-1].split(" ")
        self.SimStartIndex = int(start)
        self.SimEndIndex = int(end)
        self.SimStart = self.ordinal_to_date(self.SimStartIndex)
        self.SimEnd = self.ordinal_to_date(self.SimEndIndex)
        self.OneDTempR = 60  # in seconds

        # 1D thresholds
        min_q, min_depth = content[37][:-1].split(" ")
        min_q = float(min_q)
        min_depth = float(min_depth)

        # 1D or 2D
        model_mode, overtopping_only = content[39][:-1].split(" ")
        model_mode = int(model_mode)
        overtopping_only = int(overtopping_only)

        self.D1 = dict(
            MinQ=min_q,
            MinDepth=min_depth,
            ModelMode=model_mode,
            OvertoppingOnly=overtopping_only,
        )

        discharge_threshold = int(content[41][:-1])

        simulation_mode = int(content[43][:-1])
        results_format = int(content[45][:-1])
        calc_inundation_duration = int(content[47][:-1])
        calc_return_period = int(content[49][:-1])
        self.D2 = dict(
            DischargeThreshold=discharge_threshold,
            SimulationMode=simulation_mode,
            ResultsFormat=results_format,
            CalcInundationDuration=calc_inundation_duration,
            CalcReturnPerion=calc_return_period,
        )

    def read_xs(self, path: str):
        """read_xs.

            Read crossSections file

        read_xs method reads the cross-section data of the river and assigns it
        to an attribute "Crosssections" of type dataframe.

        Parameters
        ----------
        path: [str]
            path to the cross-section file

        Returns
        -------
        cross_sections : [dataframe]
            a dataframe attribute will be created
        """
        if self.version == 3:
            self._cross_sections = pd.read_csv(path, delimiter=",")
            self.xs_number = len(self.cross_sections)
            self.reach_ids = list(set(self.cross_sections["id"].tolist()))
        else:
            self._cross_sections = pd.read_csv(path, delimiter=",")
            self.xs_number = len(self.cross_sections)
            # TODO to be checked later now for testing of version 4
            self.xs_names = self.cross_sections["xsid"].tolist()
            self.reach_ids = list(set(self.cross_sections["id"].tolist()))

    def read_boundary_conditions(
        self,
        start: str = "",
        end: str = "",
        path: str = "",
        fmt: str = "%Y-%m-%d",
        ds: bool = False,
        dsbc_path: str = "",
    ):
        """ReadBoundaryConditions.

        Read Boundary Conditions

        ReadBoundaryConditions method reads the bc files, and since these files
        are separated, each day is written in a file, so the code is reading a
        lot of files, therefore, you can specify a specific day to start to read
        the bc H & Q from that day till the end of the simulated period

        for version 4
        - boundary condition for all points is stored in one file with the date
            stored in the first column.
        - the method will also resample the boundary condition time series to
            whatever temporal resolution you will define when instantiating the
            River object.
        - now the boundary condition is Q for the kinematic wave Approx later
            this method will be adjusted to read any boundary condition.

        Parameters
        ----------
        start: [int/str], optional
                the day you want to read the result from, the first day is 1
                not zero, you can also enter the date of the day.
                The default is ''.
        end: [int], optional
                the day you want to read the result to.
        path: [String], optional
            path to read the results from. The default is ''.
        fmt: [string]
            format of the date. fmt="%Y-%m-%d %H:%M:%S"
        ds: [bool]

        dsbc_path: [str]


        Returns
        -------
        QBC: [dataframe attribute]
            dataframe contains the Discharge boundary conditions for each day as a row and for each column are the
            "hours".
        HBC: [dataframe attribute]
            dataframe contains the water depth boundary conditions for each
            day as a row and for each column are the hours
        """
        if path != "":
            self.us_bc_path = path

        if self.version < 4:
            if start == "":
                start = 1
            if end == "":
                end = len(self.reference_index_results) - 1

            if isinstance(start, str):
                start = dt.datetime.strptime(start, fmt)
                start = np.where(self.reference_index_results == start)[0][0] + 1

            if isinstance(end, str):
                end = dt.datetime.strptime(end, fmt)
                end = np.where(self.reference_index_results == end)[0][0] + 1

            q_bc = pd.DataFrame(
                index=self.reference_index_results[start - 1 : end], columns=hours
            )
            h_bc = pd.DataFrame(
                index=self.reference_index_results[start - 1 : end], columns=hours
            )

            for i in self.days_list[start - 1 : end]:
                bc_q = np.loadtxt(
                    self.us_bc_path + str(self.id) + "-" + str(i) + ".txt",
                    dtype=np.float16,
                )
                q_bc.loc[self.reference_index.loc[i, "date"], :] = bc_q[:, 0].tolist()[
                    0 : bc_q.shape[0] : 60
                ]
                h_bc.loc[self.reference_index.loc[i, "date"], :] = bc_q[:, 1].tolist()[
                    0 : bc_q.shape[0] : 60
                ]

            self.QBC = q_bc
            self.HBC = h_bc

        else:

            def convertdate(date):
                return dt.datetime.strptime(date, fmt)

            bc = pd.read_csv(self.us_bc_path)
            bc.index = bc[bc.columns[0]].apply(convertdate)
            bc = bc.drop(bc.columns[0], axis=1)

            ind = pd.date_range(bc.index[0], bc.index[-1], freq=self.freq)
            self._usbc = pd.DataFrame(index=ind, columns=bc.columns)

            self._usbc.loc[:, :] = (
                bc.loc[:, :].resample(self.freq).mean().interpolate("linear")
            )

            if ds:
                bc = pd.read_csv(dsbc_path)
                bc.index = bc[bc.columns[0]].apply(convertdate)
                bc = bc.drop(bc.columns[0], axis=1)

                ind = pd.date_range(bc.index[0], bc.index[-1], freq=self.freq)
                self._dsbc = pd.DataFrame(index=ind, columns=bc.columns)

                self._dsbc.loc[:, :] = (
                    bc.loc[:, :].resample(self.freq).mean().interpolate("linear")
                )

    def read_sub_daily_results(
        self,
        start: str,
        end: str,
        fmt: str = "%Y-%m-%d",
        last_river_reach: bool = False,
    ):
        """read_sub_daily_results.

            Read Reach-Daily Results

            read_sub_daily_results method is used by the "sub" subclass, so most of the parameters (xs_names,
            ...) are assigned to values after reading results with other methods in the subclass

        version 4
        -

        Parameters
        ----------
        start: [string]
            DESCRIPTION.
        end: [string]
            DESCRIPTION.
        fmt: [string]
            format of the date. fmt="%Y-%m-%d %H:%M:%S"
        last_river_reach: [bool]
            True if this is the last river reach in the river network.

        Returns
        -------
        h : [dataframe]
        dataframe contains the water level time series, index is the date, and columns are the cross-section ids.
        """
        if self.version == 4:
            assert self.cross_sections, "please read the cross-sections first"

        assert isinstance(
            self.us_bc_path, str
        ), "please input the 'us_bc_path' attribute in the River or the Reach instance"

        if isinstance(start, str):
            start = dt.datetime.strptime(start, fmt)

        if isinstance(end, str):
            end = dt.datetime.strptime(end, fmt)

        ind_min = pd.date_range(start, end + dt.timedelta(days=1), freq=self.freq)[:-1]

        # how many time steps per day
        n_step = (
            len(pd.date_range(start, start + dt.timedelta(days=1), freq=self.freq)) - 1
        )

        # US boundary condition (for each day in a separate row)
        index_daily = pd.date_range(start, end + dt.timedelta(days=1), freq="D")[:-1]
        bc_q = pd.DataFrame(index=index_daily, columns=list(range(1, n_step + 1)))
        bc_h = pd.DataFrame(index=index_daily, columns=list(range(1, n_step + 1)))

        xs_name = [int(i) for i in self.xs_names]
        h = pd.DataFrame(index=ind_min, columns=xs_name)
        q = pd.DataFrame(index=ind_min, columns=xs_name)

        ii = self.date_to_ordinal(start)
        ii2 = self.date_to_ordinal(end) + 1
        list2 = list(range(ii, ii2))

        if self.version < 4:
            # read results for each day
            for i in list2:
                path = (
                    self.one_min_result_path
                    + "{0}/"
                    + str(self.id)
                    + "-{0}-"
                    + str(i)
                    + ".txt"
                )
                hh = np.transpose(np.loadtxt(path.format("h"), dtype=np.float16))
                logger.debug(path.format("h") + "- file is read")
                qq = np.transpose(np.loadtxt(path.format("q"), dtype=np.float16))
                logger.debug(path.format("q") + " file is read")
                if not last_river_reach:
                    hh = hh[:, :-1]
                    qq = qq[:, :-1]
                # add the bed level to the water depth
                hh = hh + self.cross_sections["gl"].values
                # assign the sub-daily results in the big dataframe
                ind1 = h.index[(i - list2[0]) * n_step]
                ind2 = h.index[(i - list2[0]) * n_step + n_step - 1]
                h.loc[ind1:ind2, :] = hh
                q.loc[ind1:ind2, :] = qq

                # BC
                bc = np.loadtxt(f"{self.us_bc_path}{self.id}-{i}.txt", dtype=np.float16)
                bc_q.loc[bc_q.index[i - list2[0]], :] = bc[:, 0]
                bc_h.loc[bc_h.index[i - list2[0]]] = bc[:, 1]

            self.h = h[:]
            self.q = q[:]
            self.q_bc_1min = bc_q[:]
            self.h_bc_1min = bc_h[:]
        else:
            for i in list2:
                path = f"{self.one_min_result_path}H-{str(self.ordinal_to_date(i))[:10]}.csv"
                hh = np.transpose(np.loadtxt(path, delimiter=",", dtype=np.float16))
                path = f"{self.one_min_result_path}Q-{str(self.ordinal_to_date(i))[:10]}.csv"
                qq = np.transpose(np.loadtxt(path, delimiter=",", dtype=np.float16))

                h = h + self.cross_sections["bed level"].values
                ind1 = h.index[(i - list2[0]) * len(ind_min)]
                ind2 = h.index[(i - list2[0]) * len(ind_min) + len(ind_min) - 1]
                h.loc[ind1:ind2] = hh
                q.loc[ind1:ind2] = qq

            self.h = h
            self.q = q

            # check the first day in the results and get the date of the first day and last day
            # create time series
            # TODO to be checked later now for testing
            self.from_beginning = 1  # self.results_1d['day'][0]

            self.first_day = self.ordinal_to_date(self.from_beginning)
            # if there are empty days at the beginning, the "filling missing days" is not going to detect it,
            # so ignore it here by starting from the first day in the data (data['day'][0]) dataframe
            # empty days at the beginning
            self.first_day_results = self.ordinal_to_date(self.from_beginning)
            self.last_day = self.ordinal_to_date(len(self.reference_index))

            # last days+1 as range does not include the last element
            self.days_list = list(range(self.from_beginning, len(self.reference_index)))
            self.reference_index_results = pd.date_range(
                self.first_day_results, self.last_day, freq="D"
            )

    @staticmethod
    def _read_chunks(path, chunk_size=10e5):
        """Read csv file in chunks.

        Parameters
        ----------
        path: [str]
            file path
        chunk_size: [int]
            chunk size to partition reading the file

        Returns
        -------
        DataFrame
        """
        iterator = pd.read_csv(
            path,
            header=None,
            delimiter=r"\s+",
            chunksize=chunk_size,
            iterator=True,
            compression="infer",
        )

        data = pd.DataFrame()
        cond = True

        while cond:
            try:
                chunk = iterator.get_chunk(chunk_size)
                data = pd.concat([data, chunk], ignore_index=True, sort=False)
            except StopIteration:
                cond = False

        return data

    def _read_1d_results(
        self,
        sub_id: int,
        from_day: Optional[int] = None,
        to_day: Optional[int] = None,
        path: str = None,
        fill_missing: bool = False,
        chunk_size: int = None,
        delimiter: str = r"\s+",
        extension: str = ".txt",
    ):
        r"""_read_1d_results.

            Read-1D Result Read1DResult method reads the 1D results and fill the missing days in the middle

        Parameters
        ----------
        sub_id : [integer]
            id of the sub-basin you want to read its data.
        from_day: [integer], optional
            the index of the day you want the data to start from. The default is empty.
            means read everything
        to_day: [integer], optional
            the index of the day you want the data to end to. The default is empty. which means read everything.
        path: [String], optional
            path to read the results from. The default is ''.
        fill_missing: [Bool], optional
            Fill the missing days. The default is False.
        chunk_size: [int]
            size of the chunk if you want to read the file in chunks Default is = None
        delimiter: [str]
            delimiter separating the columns in the result file. Default is r"\s+", which is a space delimiter.
        extension: [str]
            the extension of the file. Default is ".txt"

        Returns
        -------
        results_1d : [attribute]
            The results will be stored (as it is without any filter)
            in the attribute "results_1d"
        """
        # if the path is not given, try to read from the object predefined one_d_result_path
        t1 = dt.datetime.now()
        if path is None:
            path = self.one_d_result_path

        path = os.path.join(path, f"{sub_id}{extension}")

        if chunk_size is None:
            data = pd.read_csv(
                path,
                header=None,
                delimiter=delimiter,
                index_col=False,
                compression="infer",
                # engine="pyarrow"
            )
        else:
            # read the file in chunks
            data = self._read_chunks(path, chunk_size=chunk_size)

        data.columns = ["day", "hour", "xs", "q", "h", "wl"]
        days = list(set(data["day"]))
        days.sort()

        if from_day:
            if from_day not in days:
                raise ValueError(
                    f"Please use the GetDays method to select from_day:{from_day} that exist in the data"
                )
        if to_day:
            if to_day not in days:
                raise ValueError(
                    f"please use the GetDays method to select to_day: {to_day} that exist in the data"
                )

        if from_day:
            data = data.loc[data["day"] >= from_day, :]

        if to_day:
            data = data.loc[data["day"] <= to_day]
        data.reset_index(inplace=True, drop=True)

        # Cross-section data add one more xs at the end
        xs_name = self.xs_names  # + [self.xs_names[-1] + 1]
        # xs_name = data["xs"][data["day"] == data["day"][1]][data["hour"] == 1].tolist()

        if fill_missing:
            # check if there are missing days (Q was < threshold so the model didn't run)
            # fill these values with 0
            # days = list(set(data["day"]))
            # days.sort()
            missing_days = list(set(range(days[0], days[-1])) - set(days))
            if len(missing_days) > 0:
                if len(missing_days) > 10000:
                    missing_days_list = list()
                    missing_hours = list()
                    missing_xs = list()
                    for day_i in missing_days:
                        for hour_i in hours:
                            for xs_i in xs_name:
                                missing_days_list.append(day_i)
                                missing_hours.append(hour_i)
                                missing_xs.append(xs_i)
                else:
                    missing = [
                        (i, j, h) for i in missing_days for j in hours for h in xs_name
                    ]
                    missing_days_list = [i[0] for i in missing]
                    missing_hours = [i[1] for i in missing]
                    missing_xs = [i[2] for i in missing]

                missing = pd.DataFrame(
                    index=range(len(missing_days_list)), dtype=np.float64
                )
                missing["day"] = missing_days_list
                missing["hour"] = missing_hours
                missing["xs"] = missing_xs

                missing["q"] = 0
                missing["h"] = 0
                missing["wl"] = 0
                data = data.append(missing)
                # delete for the memory problem
                del missing, missing_days_list, missing_hours, missing_xs, missing_days
                data = data.sort_values(by=["day", "hour", "xs"], ascending=True)
                data.reset_index(inplace=True)

        # calculate time and print it
        t2 = dt.datetime.now()
        time_min = (t2 - t1).seconds / 60
        print(f"Time taken to read the file: {time_min:0.2f} min")
        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)
        self.results_1d = data

    @staticmethod
    def collect_1d_results(
        rdir: str,
        separate_dir_list: List[str],
        left_overtopping: bool = False,
        right_overtopping: bool = False,
        save_to: str = "",
        OneD: bool = True,
        from_file: int = None,
        to_file: int = None,
        filter_by_name: bool = False,
        delimiter: str = r"\s+",
    ):
        """Collect1DResults.

        Collect1DResults method reads the 1D separated result files and filters
        them between two numbers to remove any warmup period if exist then stack
        the result in one table then write it.

        Parameters
        ----------
        rdir : [String]
            root directory to the folder containing the separated folder.
        separate_dir_list : [List]
            list containing folder names.
        left_overtopping : [Bool]
            True if you want to combine left overtopping files.
        right_overtopping : [Bool]
            True if you want to combine right overtopping files.
        save_to: [String]
            path to the folder where data will be saved.
        OneD : [Bool]
            True if you want to combine 1D result files.
        from_file: [Integer], optional
            If the files are very big and the cache memory has a problem
            reading all the files you can specify here the order of the file
            the code will start from to combine. The default is ''.
        to_file: [Integer], optional
            If the files are very big and the cache memory has a problem
            reading all the files you can specify here the order of the file
            the code will end to combine. The default is ''.
        filter_by_name : [Bool], optional
            if the results include a wanm up period at the beginning
            or has results for some days at the end you want to filter out
            you want to include the period you want to be combined only
            in the name of the folder between () and separated with -
            ex 1d(50-80). The default is False.
            - The 50 and 80 will be used to filter the files inside the folder using df >=50, and df <=80

        delimiter: [str]
            delimeter d in the files to separate columns.

        Returns
        -------
            combined files will be written to the save_to.

        Hint
        ----
        - Make sure that files in all given directories are the same
        - The hydraulic model creates files for everything (river reaches, left overtopping, right overtopping)
            even if the overtopping files are empty.

        Examples
        --------
        project_folder/
            1d(1-7485)/
                1.txt
                1_left.txt
                1_right.txt
            1d(7485-22000)/
                1.txt
                1_left.txt
                1_right.txt
            combined/
        >>> rdir = "project_folder/"
        >>> separate_dir_list = ["1d(1-7485)", "1d(7485-22000)"]
        >>> left_overtopping = True
        >>> right_overtopping = True
        >>> save_to = "project_folder/combined/"
        >>> from_file = 1
        >>> to_file = 5
        >>> filter_by_name = True
        >>> one_d = False
        >>> River.collect_1d_results(
        >>>                       rdir, separate_dir_list, left_overtopping, right_overtopping, save_to, one_d_two_d,
        >>>                       filter_by_name=filter_by_name
        >>> )
        """
        second = "=pd.DataFrame()"
        if from_file is None:
            from_file = 0

        dir_i = separate_dir_list[0]

        if to_file is None:
            to_file = len(os.listdir(f"{rdir}/{dir_i}"))

        file_list = os.listdir(f"{rdir}/{dir_i}")[from_file:to_file]

        for j, file_i in enumerate(file_list):
            for i, dir_i in enumerate(separate_dir_list):
                logger.debug(f"Directory:{i} - {dir_i}")
                if filter_by_name:
                    try:
                        filter1 = int(dir_i.split("(")[1].split("-")[0])
                        filter2 = int(dir_i.split("(")[1].split("-")[1].split(")")[0])
                    except IndexError:
                        raise NameError(
                            f"Folder names are not the format of **(start_ind-end_ind), where start_ind and "
                            f"end_ind are integers, given folder name is {dir_i}"
                        )

                go = False
                if left_overtopping and file_i.split(".")[0].endswith("_left"):
                    logger.debug(f"Directory:{dir_i} - File:{file_i}")
                    # create data frame for the sub-basin
                    first = "L" + file_i.split(".")[0]
                    go = True

                elif right_overtopping and file_i.split(".")[0].endswith("_right"):
                    logger.debug(f"Directory:{dir_i} - File:{file_i}")
                    first = "R" + file_i.split(".")[0]
                    go = True

                # try to get the integer of the file name to make sure that it is
                # one of the 1D result files
                elif (
                    OneD
                    and not file_i.split(".")[0].endswith("_right")
                    and not file_i.split(".")[0].endswith("_left")
                ):
                    logger.debug(f"Directory:{dir_i} - File:{file_i}")
                    # create data frame for the sub-basin
                    first = "one" + file_i.split(".")[0]
                    go = True

                if go:
                    # get the updated list of variable names
                    variables = locals()

                    # read the file
                    try:
                        try:
                            temp_df = pd.read_csv(
                                f"{rdir}/{dir_i}/{file_i}",
                                header=None,
                                delimiter=delimiter,
                            )
                            # filter the data between the two dates in the folder name
                            if filter_by_name:
                                temp_df = temp_df[temp_df[0] >= filter1]
                                temp_df = temp_df[temp_df[0] <= filter2]
                            # check whether the variable exist or not
                            # if this is the first time this file's existed
                            if first not in variables.keys():
                                # create a dataframe with the name of the sub-basin.
                                total = first + second
                                exec(total)

                            # concatenate the
                            exec(first + "= pd.concat([" + first + ", temp_df])")
                        except pd.errors.EmptyDataError:
                            logger.info(f"The file: {rdir}/{dir_i}/{file_i} is empty")
                    except FileNotFoundError:
                        logger.warning(
                            f"The file: {rdir}/{dir_i}/{file_i} does not exist"
                        )
                        continue

            # Save files
            variables = list(locals().keys())
            # get sub-basins variables (starts with "One")
            var_names = ["_left", "_right", "one"]
            save_variables = [
                i
                for i in variables
                if any(i.startswith(j) or i.endswith(j) for j in var_names)
            ]
            for i, var in enumerate(save_variables):
                # var = variables[i]
                if var.endswith("_left"):
                    # put the dataframe in order first
                    exec(var + ".sort_values(by=[0,1,2], ascending=True, inplace=True)")
                    save_dir = f"{save_to}/{var[1:]}.txt"
                    exec(var + ".to_csv(save_dir, index=None, sep=' ', header=None)")
                elif var.endswith("_right"):
                    # put the dataframe in order first
                    exec(var + ".sort_values(by=[0,1,2], ascending=True, inplace=True)")
                    save_dir = f"{save_to}/{var[1:]}.txt"
                    exec(var + ".to_csv(save_dir, index=None, sep=' ', header=None)")
                elif var.startswith("one"):
                    # put the dataframe in order first
                    exec(var + ".sort_values(by=[0,1,2], ascending=True, inplace=True)")
                    logger.debug(f"Saving {var[3:]}.txt")
                    save_dir = f"{save_to}/{var[3:]}.txt"
                    exec(var + ".to_csv(save_dir, index=None, sep=' ', header=None)")
                # delete the dataframe to free memory
                exec(f"del {var}")

    @staticmethod
    def _read_rrm_results(
        version: int,
        rrm_reference_index,
        path: str,
        node_id: Union[int, str],
        from_day: int,
        to_day: int,
        date_format: str = "%d_%m_%Y",
    ) -> DataFrame:
        """ReadRRMResults.

        ReadRRMResults is a static method to read the results of the rainfall-runoff
        model

        Parameters
        ----------
        version: []

        rrm_reference_index: []

        path: [String]
            path to the result files.
        node_id: [Integer]
            the id of the given sub-basin.
        from_day: [integer], optional
                the day you want to read the result from, the first day is 1 not zero. The default is ''.
        to_day: [integer], optional
                the day you want to read the result to.
        date_format: [str]
            format of the date string

        Returns
        -------
        q: [Dataframe]
            time series of the runoff.
        """
        rpath = os.path.join(path, f"{node_id}.txt")
        if version < 3:
            q = pd.read_csv(rpath, header=None)
            q = q.rename(columns={0: node_id})
            q.index = list(range(1, len(q) + 1))

            if not from_day:
                from_day = 1
            if not to_day:
                to_day = len(q)

            q = q.loc[q.index >= from_day, :]
            q = q.loc[q.index <= to_day]
        else:
            q = pd.read_csv(rpath, header=None, skiprows=1)

            q.index = [dt.datetime.strptime(date, date_format) for date in q[0]]
            del q[0]

            # convert the date into integer index
            s = np.where(rrm_reference_index["date"] == q.index[0])[0][0] + 1
            e = np.where(rrm_reference_index["date"] == q.index[-1])[0][0] + 1
            q.index = list(range(s, e + 1))

            if not from_day:
                from_day = s
            if not to_day:
                to_day = e

            q = q.loc[q.index >= from_day, :]
            q = q.loc[q.index <= to_day, :]

            q = q[1].to_frame()
            q = q.rename(columns={1: node_id})

        return q

    def kinematic_wave(self, start: str = "", end: str = "", fmt: str = "%Y-%m-%d"):
        """kinematic_wave.

            kinematic_wave apply the kinematic wave approximation of the shallow
            water equation to the 1d river reach

        fmt: [string]
            format of the date. fmt="%Y-%m-%d %H:%M:%S"

        Returns
        -------
        Q: [array]
            discharge time series with the rows as time and columns as space
        H: [array]
            water depth time series with the rows as time and columns as space
        """
        if start == "":
            start = self.start
        else:
            start = dt.datetime.strptime(start, fmt)

        if end == "":
            end = self.end
        else:
            end = dt.datetime.strptime(end, fmt)

        # ind = pd.date_range(start, end, freq=self.freq)

        # TODO to be checked later now for testing
        # self.from_beginning = self.indsub[np.where(self.indsub == start)[0][0]]

        self.first_day = self.indsub[np.where(self.indsub == start)[0][0]]
        # if there are empty days at the beginning, the "filling missing days" is not going to detect it,
        # so ignore it here by starting from the first day in the data (data['day'][0]) dataframe
        # empty days at the beginning
        # self.first_day_results = self.indsub[np.where(self.indsub == start)[0][0]]
        self.last_day = self.indsub[np.where(self.indsub == end)[0][0]]

        # last days+1 as range does not include the last element
        # self.days_list = list(range(self.from_beginning, len(self.reference_index)))
        self.reference_index_results = pd.date_range(
            self.first_day, self.last_day, freq=self.freq
        )

        usbc = self.usbc.loc[self.reference_index_results, :]
        SaintVenant.kinematic_1d(self, usbc)

    def preissmann(
        self,
        start: str = "",
        end: str = "",
        fmt: str = "%Y-%m-%d",
        max_iteration: int = 10,
        beta: int = 1,
        epsi: float = 0.5,
        theta: float = 0.5,
    ):
        """preissmann.

            preissmann scheme solving the whole shallow water equation.

        Parameters
        ----------
        start : TYPE, optional
            DESCRIPTION. The default is ''.
        end: TYPE, optional
            DESCRIPTION. The default is ''.
        fmt: [string]
            format of the date. fmt="%Y-%m-%d %H:%M:%S"
        max_iteration: TYPE, optional
            DESCRIPTION. The default is 10.
        beta: TYPE, optional
            DESCRIPTION. The default is 1.
        epsi: TYPE, optional
            DESCRIPTION. The default is 0.5.
        theta: TYPE, optional
            DESCRIPTION. The default is 0.5.

        Returns
        -------
        None.
        """

        if start == "":
            start = self.start
        else:
            start = dt.datetime.strptime(start, fmt)

        if end == "":
            end = self.end
        else:
            end = dt.datetime.strptime(end, fmt)

        # ind = pd.date_range(start, end, freq=self.freq)

        # TODO to be checked later now for testing
        # self.from_beginning = self.indsub[np.where(self.indsub == start)[0][0]]

        self.first_day = self.indsub[np.where(self.indsub == start)[0][0]]
        # if there are empty days at the beginning, the "filling missing days" is not going to detect it,
        # so ignore it here by starting from the first day in the data (data['day'][0]) dataframe
        # empty days at the beginning
        # self.first_day_results = self.indsub[np.where(self.indsub == start)[0][0]]
        self.last_day = self.indsub[np.where(self.indsub == end)[0][0]]

        # last days+1 as range does not include the last element
        # self.days_list = list(range(self.from_beginning, len(self.reference_index)))
        self.reference_index_results = pd.date_range(
            self.first_day, self.last_day, freq=self.freq
        )

        # usbc = self.qusbc.loc[self.reference_index_results,:]
        # dsbc = self.qusbc.loc[self.reference_index_results, :]
        saintpreis = SaintVenant(
            max_iteration=max_iteration, beta=beta, epsi=epsi, theta=theta
        )
        saintpreis.preissmann(self)

    def storage_cell(self, start: str = "", end: str = "", fmt: str = "%Y-%m-%d"):
        """storage_cell.

        Parameters
        ----------
        start: [str]
            start data.
        end: [str]
            end data.
        fmt: [string]
            format of the date. fmt="%Y-%m-%d %H:%M:%S".

        Returns
        -------
        q: [array]
            discharge time series with the rows as time and columns as space
        h: [array]
            water depth time series with the rows as time and columns as space
        """
        if start == "":
            start = self.start
        else:
            start = dt.datetime.strptime(start, fmt)

        if end == "":
            end = self.end
        else:
            end = dt.datetime.strptime(end, fmt)
        # ind = pd.date_range(start, end, freq=self.freq)

        # TODO to be checked later now for testing
        # self.from_beginning = self.indsub[np.where(self.indsub == start)[0][0]]

        self.first_day = self.indsub[np.where(self.indsub == start)[0][0]]
        # if there are empty days at the beginning the filling missing days is not going to detect it
        # so ignore it here by starting from the first day in the data (data['day'][0]) dataframe
        # empty days at the beginning
        # self.first_day_results = self.indsub[np.where(self.indsub == start)[0][0]]
        self.last_day = self.indsub[np.where(self.indsub == end)[0][0]]

        # last days+1 as range does not include the last element
        # self.days_list = list(range(self.from_beginning, len(self.reference_index)))
        self.reference_index_results = pd.date_range(
            self.first_day, self.last_day, freq=self.freq
        )

        usbc = self.usbc.loc[self.reference_index_results, :]
        SaintVenant.storage_cell(self, usbc)

    def animate_flood_wave(
        self,
        start,
        end,
        interval: float = 0.00002,
        xs: int = 0,
        xs_before: int = 10,
        xs_after: int = 10,
        fmt: str = "%Y-%m-%d %H:%M:%S",
        text_location: float = 2,
        xaxis_label_size: float = 15,
        yaxis_label_size: float = 15,
        n_x_labels: float = 50,
        # plot_bankfull_depth=False,
    ):
        """animate_flood_wave.

            animate_flood_wave method animates 1d the flood wave

        Parameters
        ----------
        start : [string]
            DESCRIPTION.
        end : [string]
            DESCRIPTION.
        fmt: [string]
            format of the date. fmt="%Y-%m-%d %H:%M:%S"

        interval: []

        xs: []

        xs_before: []

        xs_after: []

        text_location: []

        xaxis_label_size: []

        yaxis_label_size: []

        n_x_labels: []

        Returns
        -------
        None.
        """
        anim = V.river1d(
            self,
            start,
            end,
            interval=interval,
            xs=xs,
            xs_before=xs_before,
            xs_after=xs_after,
            fmt=fmt,
            text_location=text_location,
            x_axis_label_size=xaxis_label_size,
            y_axis_label_size=yaxis_label_size,
            xlabels_number=n_x_labels,
        )
        return anim

    def save_result(self, path: str):  # , fmt="%.3f"):
        """SaveResult.

        Save Result

        Parameters
        ----------
        path : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        """
        for i in range(len(self.reference_index)):
            name = str(self.reference_index.loc[self.reference_index.index[i], "date"])[
                :10
            ]
            # space is rows, time is columns
            # save results of each day separately
            np.savetxt(
                f"{path}Q-{name}.csv",
                self.Q.transpose(),
                fmt="%.3f",
                delimiter=",",
            )
            np.savetxt(
                f"{path}H-{name}.csv",
                self.H.transpose(),
                fmt="%.3f",
                delimiter=",",
            )

    def read_slope(self, path: str):
        """readSlope.

            readSlope

        Parameters
        ----------
        path : [String]
            path to the Guide.csv file including the file name and extension.

        Returns
        -------
        slope: [DataFrame]
            dataframe of the boundary condition reaches that has slope
        """
        self.slope = pd.read_csv(path, delimiter=",", header=None, skiprows=1)
        self.slope.columns = ["id", "slope"]

    def return_period(self, path):
        """ReturnPeriod.

            ReturnPeriod method reads the HQ file which contains all the computational nodes
            with HQ2, HQ10, HQ100

        Parameters
        ----------
        path: [String]
            path to the HQ.csv file including the file name and extension, i.e. "HQRhine.csv".

        Returns
        -------
        RP:[data frame attribute]
            containing the river computational node and calculated return period
            for with columns ['node','HQ2','HQ10','HQ100']
        """
        self.RP = pd.read_csv(path, delimiter=",", header=None)
        self.RP.columns = ["node", "HQ2", "HQ10", "HQ100"]

    def read_river_network(self, path):
        """read_river_network.

        RiverNetwork method rad the table of each computational node followed by
        upstream and then downstream node

        Parameters
        ----------
        path: [String]
            path to the Trace.txt file including the file name and extension.

        Returns
        -------
        river-network:[data frame attribute]
            containing the river network with columns ['Subid','US','DS']
        """
        file = open(path)
        content = file.readlines()
        file.close()
        river_network = pd.DataFrame(columns=content[0][:-1].split(","))
        # all lines except the last line
        for i in range(1, len(content)):
            river_network.loc[i - 1, river_network.columns[0:2].tolist()] = [
                int(j) for j in content[i][:-1].split(",")[0:2]
            ]
            river_network.loc[i - 1, river_network.columns[2]] = [
                int(j) for j in content[i][:-1].split(",")[2:]
            ]
        # the last line does not have the \n at the end
        i = len(content) - 1
        river_network.loc[i - 1, river_network.columns[0:2].tolist()] = [
            int(j) for j in content[i][:-1].split(",")[0:2]
        ]
        river_network.loc[i - 1, river_network.columns[2]] = [
            int(j) for j in content[i].split(",")[2:]
        ]
        river_network.columns = ["No", "id", "us"]
        self.river_network = river_network[:]
        self.Segments = self.river_network["id"].tolist()

    def trace_segment(self, sub_id: int):
        """TraceSegment.

            Trace method takes sub-basin id and trace it to get the upstream and downstream computational nodes

        Parameters
        ----------
        sub_id : [int]
            subbasin id.

        Returns
        -------
        USnode : [Integer]
            the Upstream computational node from Configuration file.
        DSnode : [Integer]
            the Downstream computational node from Configuration file.

        Examples
        --------
        >>> sub_id = 42
        >>> River.trace_segment(sub_id)
        """
        us = self.river_network["us"][
            np.where(self.river_network["id"] == sub_id)[0][0]
        ]
        for i in range(len(self.river_network)):
            if id in self.river_network.loc[i, "us"]:
                ds = self.river_network.loc[i, "id"]
                break
            else:
                ds = []

        return us, ds

    def trace2(self, sub_id, us):
        """Trace2.

        trace the river network

        Parameters
        ----------
        sub_id: TYPE
            DESCRIPTION.
        us: TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        """
        # trace the given segment
        us1, _ = self.trace_segment(sub_id)
        if len(us1) > 0:
            for i in range(len(us1)):
                us.append(us1[i])
                self.trace2(us1[i], us)

    def trace(self, sub_id):
        """Trace.

        Trace method takes the id of the segment and traces it upstream.

        Parameters
        ----------
        sub_id: TYPE
            DESCRIPTION.

        Returns
        -------
        us: [list attribute], optional
            the id of all the upstream reaches are going to be stored in a list
            attribute.
        """
        self.US = []
        self.trace2(sub_id, self.US)

    def statistical_properties(self, path: str, distribution: str = "GEV"):
        """StatisticalProperties.

            StatisticalProperties method reads the parameters of the distribution and calculates the 2, 5, 10, 15,
            20, 50, 100, 200, 500, 1000 return period discharge for each sub-basin to create the parameter file use
            the code StatisticalProperties in the 07ReturnPeriod folder

        Parameters
        ----------
        path : [String]
            path to the "Statistical Properties.txt" file including the
            file name and extention "path/Statistical Properties.txt".
            >>> "Statistical Properties.txt"
            >>> id, c, loc, scale, D-static, P-Value
            >>> 23800100,-0.0511,115.9465,42.7040,0.1311,0.6748
            >>> 23800500,0.0217,304.8785,120.0510,0.0820,0.9878
            >>> 23800690,0.0215,455.4108,193.4242,0.0656,0.9996

        distribution: [str]
            The distribution used to fit the data. Default is "GEV".

        Returns
        -------
        SP: [data frame attribute]
            containing the river computational nodes US of the sub-basins and estimated gumbel distribution
            parameters that fit the time series ['node','HQ2','HQ10','HQ100']

        Examples
        --------
        >>> import serapis.river as R
        >>> HM = R.River('Hydraulic model')
        >>> stat_properties_path = "path/to/results/statistical analysis/distribution-properties.csv"
        >>> HM.statistical_properties(stat_properties_path)
        >>> HM.SP
                       id       c        loc      scale  D-static  P-Value
        0        23800100 -0.0511   115.9465    42.7040    0.1311   0.6748
        1        23800500  0.0217   304.8785   120.0510    0.0820   0.9878
        2        23800690  0.0215   455.4108   193.4242    0.0656   0.9996
        3        23700200  0.1695  2886.8037   900.1678    0.0820   0.9878
        """
        self.SP = pd.read_csv(path, delimiter=",")
        # calculate the 2, 5, 10, 15, 20 return period discharges.
        T = np.array([2, 5, 10, 15, 20, 50, 100, 200, 500, 1000, 5000])
        self.SP = self.SP.assign(
            RP2=0,
            RP5=0,
            RP10=0,
            RP15=0,
            RP20=0,
            RP50=0,
            RP100=0,
            RP200=0,
            RP500=0,
            RP1000=0,
            RP5000=0,
        )
        cdf = 1 - (1 / T)
        dist = Distributions(distribution)
        for i in range(len(self.SP)):
            if self.SP.loc[i, "loc"] != -1:
                col1 = self.SP.columns.to_list().index("RP2")
                parameters = {
                    "loc": self.SP.loc[i, "loc"],
                    "scale": self.SP.loc[i, "scale"],
                }
                if distribution == "GEV":
                    parameters["shape"] = self.SP.loc[i, "c"]

                self.SP.loc[
                    i, self.SP.keys()[col1:].tolist()
                ] = dist.theoretical_estimate(parameters, cdf)

    def get_return_period(
        self,
        sub_id: int,
        q: Union[float, int, np.ndarray, list],
        distribution: str = "GEV",
    ):
        """get_return_period.

            get_return_period method takes given discharge and using the distribution properties of a particular
            sub-basin it calculates the return period

        Parameters
        ----------
        sub_id: [Integer]
            sub-basin id.
        q: [Float]
            Discharge value.
        distribution: [str]
            the distribution used to fit the discharge data. Default is "GEV".

        Returns
        -------
        return period:[Float]
            return period calculated for the given discharge using the parameters of the distribution for the catchment.
        """

        if not isinstance(self.SP, DataFrame):
            raise ValueError(
                "Please read the statistical properties file for the catchment first"
            )

        dist = Distributions(distribution)

        try:
            loc = np.where(self.SP["id"] == sub_id)[0][0]
            parameter = dict(
                loc=self.SP.loc[loc, "loc"], scale=self.SP.loc[loc, "scale"]
            )

            if distribution == "GEV":
                parameter["shape"] = self.SP.loc[loc, "c"]

            rp = dist.get_rp(parameter, q)

            return rp
        except IndexError:
            return -1

    def get_q_for_return_period(self, sub_id, return_period, distribution: str = "GEV"):
        """GetQForReturnPeriod.

            get the corresponding discharge to a specific return period

        Parameters
        ----------
        sub_id: [int]
            DESCRIPTION.
        return_period: TYPE
            DESCRIPTION.
        distribution: [str]
            statistical distribution. Default is "GEV"

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        if not isinstance(self.SP, DataFrame):
            raise ValueError(
                "Please read the statistical properties file for the catchment first"
            )

        if "id" not in self.SP.columns:
            raise ValueError(
                "the SP dataframe should have a column 'id' containing the id of the gauges"
            )

        cdf = 1 - (1 / return_period)
        dist = Distributions(distribution)

        try:
            loc = np.where(self.SP["id"] == sub_id)[0][0]
            parameter = dict(
                loc=self.SP.loc[loc, "loc"], scale=self.SP.loc[loc, "scale"]
            )

            if distribution == "GEV":
                parameter["shape"] = self.SP.loc[loc, "c"]

            q = dist.theoretical_estimate(parameter, cdf)
            return q
        except:
            return -1

    def get_bankfull_depth(self, function: Callable, column_name: str):
        """GetBankfullDepth.

            get_bankfull_depth method takes a function that calculates the bankfull_depth depth as a function of bankfull_depth
            width and calculates the depth.

        Parameters
        ----------
        function: [function]
            function that takes one input and calculates the depth.
        column_name: [str]
            A name for the column to store the calculated depth at the cross-section dataframe.

        Returns
        -------
        dataframe:
            column in the cross-section attribute with the calculated depth.

        Example
        -------
        >>> river = River('Rhine')
        >>> river.read_xs(xs_file)
        >>> def bankfull_depth(b):
        >>>     return 0.6354 * (b/0.7093)**0.3961
        >>> River.get_bankfull_depth(bankfull_depth, 'dbf2')
        """
        if not hasattr(self, "cross_sections"):
            raise AttributeError(
                "cross-section does not exist, please read the river cross-section first"
            )
        self.cross_sections[column_name] = (
            self.cross_sections["b"].to_frame().applymap(function)
        )

    def get_river_capacity(
        self, column_name: str, option: int = 1, distribution: str = "GEV"
    ):
        """get_river_capacity.

            GetCapacity method calculates the discharge that is enough to fill the cross-section using kinematic wave
            approximation (using a bed slope with the manning equation)

            In order to calculate the return period corresponding to each cross-section discharge, each cross-section
            needs to be assigned the id of a specific gauge, as the statistical analysis is being done for the gauges
            only, so the distribution parameters are estimated only for the gauges.

        Parameters
        ----------
        column_name : [String]
            A name for the column to store the calculated depth at the cross-section dataframe.
        option : [Integer]
            1 if you want to calculate the capacity of the bankfull area,
            2 if you want to calculate the capacity of the whole cross-section to the level of the lowest dike
        distribution: [str]
            The distribution you used to fit the discharge data. Default is GEV

        Returns
        -------
        the cross_sections dataframe will be updated with the following columns.

        Discharge: [dataframe column]
            the calculated discharge will be stored in the cross_sections
            attribute in the River object in a columns with the given column_name
        column_name+"RP":[dataframe column]
            if you already rad the statistical properties another column containing
            the corresponding return period to the discharge,
            the calculated return period will be stored in a column with a name
            the given column_name+"RP", if the column_name was QC then the discharge
            will be in a Qc columns and the return period will be in QcRP column
        """
        for i in range(len(self.cross_sections) - 1):
            # get the slope
            if self.cross_sections.loc[i, "id"] == self.cross_sections.loc[i + 1, "id"]:
                slope = (
                    self.cross_sections.loc[i, "gl"]
                    - self.cross_sections.loc[i + 1, "gl"]
                ) / self.dx
            else:
                slope = (
                    abs(
                        self.cross_sections.loc[i, "gl"]
                        - self.cross_sections.loc[i - 1, "gl"]
                    )
                    / self.dx
                )
            self.cross_sections.loc[i, "Slope"] = slope

            if option == 1:
                # bankfull area
                self.cross_sections.loc[i, column_name] = (
                    (1 / self.cross_sections.loc[i, "m"])
                    * self.cross_sections.loc[i, "b"]
                    * (self.cross_sections.loc[i, "dbf"]) ** (5 / 3)
                )
                self.cross_sections.loc[i, column_name] = self.cross_sections.loc[
                    i, column_name
                ] * slope ** (1 / 2)

            else:
                # the lowest dike.
                # get the vortices of the cross-sections.
                h = self.cross_sections.loc[i, ["zl", "zr"]].min()
                hl, hr, bl, br, b, dbf = self.cross_sections.loc[
                    i, ["hl", "hr", "bl", "br", "b", "dbf"]
                ].tolist()
                bed_level = self.cross_sections.loc[i, "gl"]
                coords = self.get_vortices(h - bed_level, hl, hr, bl, br, b, dbf)
                # get the area and perimeters
                area, perimeter = self.polygon_geometry(coords)
                # self.cross_sections.loc[i,'area'] = area
                # self.cross_sections.loc[i,'perimeter'] = perimeter
                self.cross_sections.loc[i, column_name] = (
                    (1 / self.cross_sections.loc[i, "m"])
                    * area
                    * ((area / perimeter) ** (2 / 3))
                )
                self.cross_sections.loc[i, column_name] = self.cross_sections.loc[
                    i, column_name
                ] * slope ** (1 / 2)

            if isinstance(self.SP, DataFrame):
                if "gauge" not in self.cross_sections.columns.tolist():
                    raise ValueError(
                        "To calculate the return period for each cross-section, a column with "
                        "the corresponding gauge-id should be in the cross-section file"
                    )
                rp = self.get_return_period(
                    self.cross_sections.loc[i, "gauge"],
                    self.cross_sections.loc[i, column_name],
                    distribution=distribution,
                )
                if np.isnan(rp):
                    rp = -1
                self.cross_sections.loc[i, column_name + "RP"] = round(rp, 2)

    def calibrate_dike(
        self,
        objective_return_period: Union[str, int],
        current_return_period: Union[str, int],
    ):
        """calibrate_dike.

            calibrate_dike method takes cross-section, and based on a given return period, it calcultes a new dimension.

        Parameters
        ----------
        objective_return_period: [string]
            Column name in the cross-section dataframe you have to enter it.
        current_return_period: [string]
            Column name in the cross-section dataframe created by the GetCapacity method.

        Returns
        -------
        New rp: [data frame column]
            column in the cross-section dataframe containing the new return period.
        New Capacity: [data frame column]
            column in the cross-section dataframe containing the enough discharge to fill the cross-section after
            raising the dikes to the objective return period.
        """
        if not isinstance(self.SP, DataFrame):
            raise TypeError(
                "Please read the statistical properties file first using StatisticalProperties method"
            )

        if not isinstance(self.cross_sections, DataFrame):
            raise TypeError(
                "please read the cross-section data first with the method CrossSections"
            )

        if current_return_period not in self.cross_sections.columns:
            raise ValueError(
                f"{current_return_period} in not in the cross section data please use GetCapacity method to "
                f"calculate the current return period"
            )

        if objective_return_period not in self.cross_sections.columns:
            raise ValueError(
                f"{objective_return_period} in not in the cross-section data please create a column in the cross "
                "section data containing the objective return period"
            )

        self.cross_sections.loc[:, "zlnew"] = self.cross_sections.loc[:, "zl"]
        self.cross_sections.loc[:, "zrnew"] = self.cross_sections.loc[:, "zr"]

        for i in range(len(self.cross_sections) - 2):
            if self.cross_sections.loc[i, "id"] == self.cross_sections.loc[i + 1, "id"]:
                slope = (
                    self.cross_sections.loc[i, "gl"]
                    - self.cross_sections.loc[i + 1, "gl"]
                ) / self.dx
            else:
                slope = (
                    abs(
                        self.cross_sections.loc[i, "gl"]
                        - self.cross_sections.loc[i - 1, "gl"]
                    )
                    / self.dx
                )
            # self.cross_sections.loc[i,'Slope'] = slope
            hl, hr, bl, br, b, dbf = self.cross_sections.loc[
                i, ["hl", "hr", "bl", "br", "b", "dbf"]
            ].tolist()
            bed_level = self.cross_sections.loc[i, "gl"]

            # compare the current return period with the desired return period.
            if (
                self.cross_sections.loc[i, current_return_period]
                < self.cross_sections.loc[i, objective_return_period]
                and self.cross_sections.loc[i, current_return_period] != -1
            ):
                logger.debug("XS-" + str(self.cross_sections.loc[i, "xsid"]))
                logger.debug(
                    "Old rp = " + str(self.cross_sections.loc[i, current_return_period])
                )
                logger.debug(
                    "Old h = " + str(self.cross_sections.loc[i, ["zl", "zr"]].min())
                )

                self.cross_sections.loc[i, "New rp"] = self.cross_sections.loc[
                    i, current_return_period
                ]

                while (
                    self.cross_sections.loc[i, "New rp"]
                    < self.cross_sections.loc[i, objective_return_period]
                ):
                    # get the vortices of the cross-sections
                    if (
                        self.cross_sections.loc[i, "zlnew"]
                        < self.cross_sections.loc[i, "zrnew"]
                    ):
                        self.cross_sections.loc[i, "zlnew"] = (
                            self.cross_sections.loc[i, "zlnew"] + 0.1
                        )
                    else:
                        self.cross_sections.loc[i, "zrnew"] = (
                            self.cross_sections.loc[i, "zrnew"] + 0.1
                        )

                    h = self.cross_sections.loc[i, ["zlnew", "zrnew"]].min()
                    coords = self.get_vortices(h - bed_level, hl, hr, bl, br, b, dbf)
                    # get the area and perimeters
                    area, perimeter = self.polygon_geometry(coords)
                    self.cross_sections.loc[i, "New Capacity"] = (
                        (1 / self.cross_sections.loc[i, "m"])
                        * area
                        * ((area / perimeter) ** (2 / 3))
                    )
                    self.cross_sections.loc[
                        i, "New Capacity"
                    ] = self.cross_sections.loc[i, "New Capacity"] * slope ** (1 / 2)

                    rp = self.get_return_period(
                        self.cross_sections.loc[i, "gauge"],
                        self.cross_sections.loc[i, "New Capacity"],
                    )

                    self.cross_sections.loc[i, "New rp"] = round(rp, 2)

                logger.info(f"New rp = {round(rp, 2)}")
                logger.info(f"New h = {round(h, 2)}")
                logger.info("---------------------------")

    def parse_overtopping(
        self, overtopping_result_path: str = None, delimiter: str = r"\s+"
    ):
        r"""parse_overtopping.

            Overtopping method reads the overtopping files, and for each cross-section in each sub-basin it will
            strore the days where overtopping happens in this cross-section.

            you do not need to delete empty files or anything just give the code the sufix you used for the left
            overtopping file and the sufix you used for the right overtopping file.

        Parameters
        ----------
        overtopping_result_path: [str]
            a path to the folder including 2D results.
        delimiter: [str]
            Default is r"\s+".

        Returns
        -------
        overtopping_reaches_left: [dictionary attribute]
            dictionary having sub-basin ids as a key, and for each sub-basin, it contains dictionary for each cross-
            section having the days of overtopping.
            {reach_1: {xs_1: [day1, day2, ....]}}
            >>> {
            >>>     '10': {
            >>>             17466: [10296, 10293, 10294, 10295],
            >>>             17444: [10292, 10293, 10294, 10295, 10296, 5918, 5919]
            >>>          },
            >>>     '11': {17669: [10292, 10293, 10294, 10295, 10296]},
            >>>     '12': {
            >>>             17692: [10292, 10293, 10294, 10295, 10296, 5918, 5919],
            >>>             17693: [10296, 10293, 10294, 10295],
            >>>             17694: [10292, 10293, 10294, 10295, 10296]
            >>>          },
            >>>     '13': {17921: [8200, 8201, 8199], 17922: [8200, 8201, 8198, 8199]},
            >>>     '15': {18153: [8200, 8201, 8199]
            >>> }
        overtopping_reaches_right : [dictionary attribute]
            dictionary having sub-basin ids as a key and for each sub-basins, it contains dictionary for each
            cross-section having the days of overtopping.
        """
        # sort files
        left_overtopping = list()
        right_overtopping = list()
        # get names of files that have _left or _right at its end
        if overtopping_result_path is None:
            overtopping_result_path = self.one_d_result_path

        all_1d_files = os.listdir(overtopping_result_path)
        for i in range(len(all_1d_files)):
            if all_1d_files[i].endswith(self.left_overtopping_suffix):
                left_overtopping.append(all_1d_files[i])
            if all_1d_files[i].endswith(self.right_overtopping_suffix):
                right_overtopping.append(all_1d_files[i])

        # two dictionaries for overtopping left and right
        overtopping_subs_left = dict()
        overtopping_subs_right = dict()
        # the _left and _right files has all the overtopping discharge.
        # but sometimes the sum of all the overtopping is less than a threshold specified.
        # and then the 2D algorithm does not run, so these cross sections you will not find
        # any inundation beside it in the maps, but you will find it in the _left or _right maps

        # for each sub-basin that has overtopping from the left dike
        for i in range(len(left_overtopping)):
            # open the file (if there are no columns, the file is empty)
            data = pd.read_csv(
                rf"{overtopping_result_path}/{left_overtopping[i]}",
                header=None,
                delimiter=delimiter,
            )
            # add the sub-basin to the overtopping dictionary of sub-basins
            overtopping_subs_left[
                left_overtopping[i][: -len(self.left_overtopping_suffix)]
            ] = dict()

            # get the XS that overtopping happened from
            xs = list(set(data.loc[:, 2]))
            # for each XS get the days
            for j in range(len(xs)):
                overtopping_subs_left[
                    left_overtopping[i][: -len(self.left_overtopping_suffix)]
                ][xs[j]] = list(set(data[0][data[2] == xs[j]].tolist()))

        for i in range(len(right_overtopping)):
            data = pd.read_csv(
                rf"{overtopping_result_path}/{right_overtopping[i]}",
                header=None,
                delimiter=delimiter,
            )
            # add the sub-basin to the overtopping dictionary of sub-basins
            overtopping_subs_right[
                right_overtopping[i][: -len(self.right_overtopping_suffix)]
            ] = dict()

            # get the XS that overtopping happened from
            xs = list(set(data.loc[:, 2]))
            # for each XS get the days
            for j in range(len(xs)):
                overtopping_subs_right[
                    right_overtopping[i][: -len(self.right_overtopping_suffix)]
                ][xs[j]] = list(set(data[0][data[2] == xs[j]].tolist()))

        self.overtopping_reaches_left = overtopping_subs_left
        self.overtopping_reaches_right = overtopping_subs_right

    def get_overtopped_xs(self, day: int, all_event_days=True):
        """get_overtopped_xs.

            - get_overtopped_xs method get the cross sections that was overtopped in a given date
            - you have to read the overtopping data first with the method parse_overtopping
            - since inudation maps gets the max depth for the whole event, the method can also trace the event back
            to the beginning and get all the overtopped XS from the beginning of the Event till the given day
            - you have to give the River object the event_index attribute from the Event Object.

        Parameters
        ----------
        day: [int]
            the day you want to get the overtopped cross-section for.
        all_event_days: [Bool], optional
            if you want to get the overtopped cross-section for this day only or for the whole event. The default is
            True.
            - if True, the function traces from the given day back till the beginning of the event.

        Returns
        -------
        xs_left : [list]
            list of cross-section ids that has overtopping from the left bank.
        xs_right : [list]
            list of cross-section ids that has overtopping from the right bank.

        Example
        -------
        - for a given day
        >>> river = River('Rhine')
        >>> river.Overtopping("/results/1d/")
        >>> day = 1122
        >>> xs_left, xs_right = river.get_overtopped_xs(day,False)

        - from the beginning of the event till the given day
        >>> river = River('Rhine')
        >>> river.create_from_overtopping("/results/1d/")
        - read precreated event_index table
        >>> river.read_event_index("event_index.txt")
        - give the event_index table to the River Object
        >>> river.event_index = river.event_index
        >>> day = 1122
        >>> xs_left, xs_right = river.get_overtopped_xs(day, False)
        """
        if not hasattr(self, "event_index"):
            raise ValueError(
                "event_index does not exist, please read the event_index first."
            )

        if all_event_days:
            # first get event index
            loc = np.where(self.event_index["id"] == day)[0][0]
            ind = self.event_index.loc[loc, "index"]
            event_days = self.event_index.loc[
                self.event_index["index"] == ind, "id"
            ].values.tolist()
            # as there might be gaps in the middle, get the first and last day and generate a list of all days.
            event_days = list(range(event_days[0], event_days[-1] + 1))
        else:
            event_days = [day]

        xs_left = list()
        xs_right = list()

        for day_i in event_days:
            # for each river reach in the overtopping left dict
            for reach_i in self.overtopping_reaches_left.keys():
                # get all cross-sections that were overtopped before.
                xs = list(self.overtopping_reaches_left[reach_i].keys())
                # for each cross-section check if the day is sored inside.
                for xs_i in xs:
                    if day_i in self.overtopping_reaches_left[reach_i][xs_i]:
                        xs_left.append(xs_i)

            for reach_i in self.overtopping_reaches_right.keys():
                xs = list(self.overtopping_reaches_right[reach_i].keys())

                for xs_i in xs:
                    if day_i in self.overtopping_reaches_right[reach_i][xs_i]:
                        xs_right.append(xs_i)

        xs_left = list(set(xs_left))
        xs_right = list(set(xs_right))

        return xs_left, xs_right

    def get_sub_basin(self, xsid: int):
        """get_sub_basin.

            get_sub_basin method returns the sub-basin that the cross-section belongs to.

        Parameters
        ----------
        xsid: [Integer]
            cross-section id.

        Returns
        -------
        Integer:
                sub-basin id.
        """
        loc = np.where(self.cross_sections["xsid"] == xsid)[0][0]
        return self.cross_sections.loc[loc, "id"]

    def get_flooded_reaches(
        self, overtopped_xs: List = None, day: List[int] = None, all_event_days=True
    ):
        """get_flooded_reaches.

            get_flooded_reaches gets the inundeated sub-basins.

        Parameters
        ----------
        overtopped_xs: [list], optional
            list of cross-sections overtopped (if you already used the GetOvertoppedXS method to get the overtopped
            XSs for a specific day). The default is []. If entered, the algorithm is not going to look at the over
            arguments of the method.
        day: [list], optional
            if you want to get the flooded subs for a specific list of days. The default is 1.
        all_event_days: [Bool], optional in case user entered overtopped_xs
            if the user entered day, the all_event_days should be given. The default is True.

        Returns
        -------
        Subs : TYPE
            DESCRIPTION.

        Examples
        --------
        - get the flooded subs for a specific days
            >>> floodedSubs = River.get_flooded_reaches(day = [1122,1123], all_event_days=False)

        - get the flooded subs from already obtained overtopped XSs
            >>> day = 1122
            >>> xs_left, xs_right = River.get_overtopped_xs(day, False)
            >>> floodedSubs = River.get_flooded_reaches(overtopped_xs = xs_left + xs_right, all_event_days=False)
        """
        if overtopped_xs is None:
            overtopped_xs = []
            if day is None:
                raise ValueError(
                    "You have to enter the overtopped_xs or the day, both are given None"
                )

        reaches = []
        # if you already used the GetOvertoppedXS and have a list of xs overtopped at specific day.
        if len(overtopped_xs) > 0:
            overtopped_xs = list(set(overtopped_xs))
            for xs_i in overtopped_xs:
                reaches.append(self.get_sub_basin(xs_i))
        else:
            for day_i in day:
                xs_left, xs_right = self.get_overtopped_xs(day_i, all_event_days)
                overtopped_xs = xs_left + xs_right
                overtopped_xs = list(set(overtopped_xs))

                for xs_i in overtopped_xs:
                    reaches.append(self.get_sub_basin(xs_i))

        # to remove duplicate subs
        reaches = list(set(reaches))
        return reaches

    def _get_detailed_overtopping(
        self,
        flooded_reaches: List[int],
        event_days: List[int],
        left: bool,
        delimiter: str,
    ) -> DataFrame:
        columns = flooded_reaches + ["sum"]
        detailed_overtopping = pd.DataFrame(index=event_days + ["sum"], columns=columns)

        if left:
            suffix = self.left_overtopping_suffix
        else:
            suffix = self.right_overtopping_suffix

        for i, reach_i in enumerate(flooded_reaches):
            data = pd.read_csv(
                rf"{self.one_d_result_path}/{reach_i}{suffix}",
                header=None,
                delimiter=delimiter,
            )
            # get the days in the sub
            days = list(set(data.loc[:, 0]))
            for j, day_i in enumerate(event_days):
                # check whether this sub-basin has flooded in this particular day
                if day_i in days:
                    # filter the dataframe to the discharge column (3) and the days
                    detailed_overtopping.loc[day_i, reach_i] = data.loc[
                        data[0] == day_i, 3
                    ].sum()
                else:
                    detailed_overtopping.loc[day_i, reach_i] = 0

            return detailed_overtopping

    def detailed_overtopping(
        self, flooded_reaches: List[int], event_days: List[int], delimiter: str = r"\s+"
    ):
        r"""detailed_overtopping.

            detailed_overtopping method takes a list of days and the flooded subs-basins in those days and get the left
            and right overtopping for each sub-basin for each day.

        Parameters
        ----------
        flooded_reaches : [list]
            list of sub-basins that are flooded during the given days.
        event_days : [list]
            list of days of an event.
        delimiter: str
            Delimiter used in the 1D result files. Default is r"\s+".

        Returns
        -------
        detailed_overtopping_left: [dataframe attribute]
            dataframe having for each day of the event the left overtopping to each sub-basin.
                         5       sum
            8195     892.0     892.0
            8196   20534.7   20534.7
            8197   66490.8   66490.8
            8198   99162.4   99162.4
            8199  129359.3  129359.3
            8200  123513.0  123513.0
            sum   439952.2       NaN

        detailed_overtopping_right: [dataframe attribute]
            dataframe having for each day of the event the right overtopping to each sub-basin.
        """
        self.detailed_overtopping_left = self._get_detailed_overtopping(
            flooded_reaches, event_days, left=True, delimiter=delimiter
        )

        self.detailed_overtopping_right = self._get_detailed_overtopping(
            flooded_reaches, event_days, left=False, delimiter=delimiter
        )

        # sum overtopping for each day
        for j, day_i in enumerate(event_days):
            self.detailed_overtopping_left.loc[
                day_i, "sum"
            ] = self.detailed_overtopping_left.loc[day_i, :].sum()
            self.detailed_overtopping_right.loc[
                day_i, "sum"
            ] = self.detailed_overtopping_right.loc[day_i, :].sum()
        # sum overtopping for each subbasin.
        for j, reach_i in enumerate(flooded_reaches):
            self.detailed_overtopping_left.loc[
                "sum", reach_i
            ] = self.detailed_overtopping_left.loc[:, reach_i].sum()
            self.detailed_overtopping_right.loc[
                "sum", reach_i
            ] = self.detailed_overtopping_right.loc[:, reach_i].sum()

        # self.detailed_overtopping_left.loc['sum','sum'] = self.detailed_overtopping_left.loc[:,'sum'].sum()
        # self.detailed_overtopping_right.loc['sum','sum'] = self.detailed_overtopping_right.loc[:,'sum'].sum()

    def coordinates(self, bankfull_depth: bool = False):
        """coordinates.

            the `coordinates` method calculates the real coordinates for all the Vortices of the cross-section.

        Parameters
        ----------
        bankfull_depth: [Bool], optional
            if the cross-section data has a bankfull depth or not. The default is False.

        Returns
        -------
        the coordenates will be added to the "cross_sections" attribute.
        """
        if bankfull_depth:
            self.cross_sections = self.cross_sections.assign(
                x1=0,
                y1=0,
                z1=0,
                x2=0,
                y2=0,
                z2=0,
                x3=0,
                y3=0,
                z3=0,
                x4=0,
                y4=0,
                z4=0,
                x5=0,
                y5=0,
                z5=0,
                x6=0,
                y6=0,
                z6=0,
                x7=0,
                y7=0,
                z7=0,
                x8=0,
                y8=0,
                z8=0,
            )

            for i in range(len(self.cross_sections)):
                inputs = self.cross_sections.loc[
                    i, list(self.cross_sections.columns)[3:15]
                ].tolist()
                dbf = self.cross_sections.loc[i, list(self.cross_sections.columns)[16]]

                outputs = self.get_coordinates(inputs, dbf)

                self.cross_sections.loc[
                    i, ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
                ] = outputs[0]

                self.cross_sections.loc[
                    i, ["y1", "y2", "y3", "y4", "y5", "y6", "y7", "y8"]
                ] = outputs[1]

                self.cross_sections.loc[
                    i, ["z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8"]
                ] = outputs[2]
        else:
            self.cross_sections = self.cross_sections.assign(
                x1=0,
                y1=0,
                z1=0,
                x2=0,
                y2=0,
                z2=0,
                x3=0,
                y3=0,
                z3=0,
                x4=0,
                y4=0,
                z4=0,
                x5=0,
                y5=0,
                z5=0,
                x6=0,
                y6=0,
                z6=0,
            )
            dbf = False
            for i in range(len(self.cross_sections)):
                inputs = self.cross_sections.loc[
                    i, list(self.cross_sections.columns)[3:15]
                ].tolist()

                outputs = self.get_coordinates(inputs, dbf)

                self.cross_sections.loc[
                    i, ["x1", "x2", "x3", "x4", "x5", "x6"]
                ] = outputs[0]

                self.cross_sections.loc[
                    i, ["y1", "y2", "y3", "y4", "y5", "y6"]
                ] = outputs[1]

                self.cross_sections.loc[
                    i, ["z1", "z2", "z3", "z4", "z5", "z6"]
                ] = outputs[2]

        # TODO create a method to take the created coordinates and convert each cross section
        # into  a polygon
        # TODO another method to take the cross section coordinates of a whole sub basins
        # and convert them into one polygon

    # def CreatePolygons(self):

    @staticmethod
    def polygon_geometry(coordinates: np.ndarray):
        """PolygonGeometry.

            PolygonGeometry method calculates the area and perimeter of some coordinates

        Parameters
        ----------
        coordinates: [array]
            numpy array in the shape of (n*2) where n is the number of points.

        Returns
        -------
        area: [float]
            area between the coordinates.
        peri: [float]
            perimeter between the coordinates.

        Example
        -------
            coords = np.array([[0,1],[0,0],[5,0],[5,1]])
            RV.River.PolygonGeometry(coords)
        """
        area = 0.0
        peri = 0.0
        for i in range(np.shape(coordinates)[0] - 1):
            area = (
                area
                + coordinates[i, 0] * coordinates[i + 1, 1]
                - coordinates[i + 1, 0] * coordinates[i, 1]
            )
            peri = (
                peri
                + (
                    (coordinates[i + 1, 0] - coordinates[i, 0]) ** 2
                    + (coordinates[i + 1, 1] - coordinates[i, 1]) ** 2
                )
                ** 0.5
            )

        area = (
            area
            + coordinates[np.shape(coordinates)[0] - 1, 0] * coordinates[0, 1]
            - coordinates[0, 0] * coordinates[np.shape(coordinates)[0] - 1, 1]
        )
        area = area * 0.5

        return area, peri

    @staticmethod
    def poly_area(coordinates: np.ndarray):
        """poly_area.

            poly_area method calculates the area between given coordinates.

        Parameters
        ----------
        coordinates: [array]
            numpy array in the shape of (n*2) where n is the number of points.

        Returns
        -------
        area: [float]
            area between the coordinates.

        Example:
        -------
        coords = np.array([[0,1],[0,0],[5,0],[5,1]])
        River.PolyArea(coords)
        """
        coordinates = np.array(coordinates)
        area = 0.0
        for i in range(np.shape(coordinates)[0] - 1):
            # cross multiplication
            area = (
                area
                + coordinates[i, 0] * coordinates[i + 1, 1]
                - coordinates[i + 1, 0] * coordinates[i, 1]
            )
        area = (
            area
            + coordinates[np.shape(coordinates)[0] - 1, 0] * coordinates[0, 1]
            - coordinates[0, 0] * coordinates[np.shape(coordinates)[0] - 1, 1]
        )
        area = area * 0.5

        return area

    @staticmethod
    def poly_perimeter(coordinates: np.ndarray):
        """poly_perimeter.

            poly_perimeter method calculates the perimeter between given coordinates.

        Parameters
        ----------
        coordinates: [array]
            numpy array in the shape of (n*2) where n is the number of points

        Returns
        -------
        peri: [float]
            perimeter between the coordinates.
        """
        peri = 0.0
        for i in range(np.shape(coordinates)[0] - 1):
            # next point coord - current point coord
            peri = (
                peri
                + (
                    (coordinates[i + 1, 0] - coordinates[i, 0]) ** 2
                    + (coordinates[i + 1, 1] - coordinates[i, 1]) ** 2
                )
                ** 0.5
            )

        return peri

    @staticmethod
    def get_coordinates(xs_geometry: list, dbf: float = None):
        """get_coordinates.

            get_coordinates calculates the coordinates of all the points (vortices) of the cross-section.

        Parameters
        ----------
        xs_geometry: [list]
            1- bed_level: [Float]
                DESCRIPTION.
            2- bank_left_level: [Float]
                DESCRIPTION.
            3- bank_right_level: [Float]
                DESCRIPTION.
            4- inter_pl_height: [Float]
                Intermediate point left.
            5- inter_pr_height: [Float]
                Intermediate point right.
            6- bl: [Float]
                DESCRIPTION.
            7- br: [Float]
                DESCRIPTION.
            8- xl: [Float]
                DESCRIPTION.
            9- yl: [Float]
                DESCRIPTION.
            10- xr: [Float]
                DESCRIPTION.
            11- yr: [Float]
                DESCRIPTION.
            12- b: [Float]
                DESCRIPTION.
            13- dbf: [Float/Bool]
                DESCRIPTION.
        dbf: [float]
            bankfull depth. Default is None.

        Returns
        -------
        x_coords: [List]
            DESCRIPTION.
        x_coords: [List]
            DESCRIPTION.
        z_coords: [List]
            DESCRIPTION.
        """
        bed_level = xs_geometry[0]
        bank_left_level = xs_geometry[1]
        bank_right_level = xs_geometry[2]
        inter_pl_height = xs_geometry[3]
        inter_pr_height = xs_geometry[4]
        bl = xs_geometry[5]
        br = xs_geometry[6]
        xl = xs_geometry[7]
        yl = xs_geometry[8]
        xr = xs_geometry[9]
        yr = xs_geometry[10]
        b = xs_geometry[11]

        x_coords = list()
        y_coords = list()
        z_coords = list()
        # point 1
        x_coords.append(xl)
        y_coords.append(yl)
        z_coords.append(bank_left_level)
        # 8 points cross sections
        if dbf:
            # point 2
            x_coords.append(xl)
            y_coords.append(yl)
            z_coords.append(bed_level + dbf + inter_pl_height)
            # point 3
            x_coords.append(xl + (bl / (bl + br + b)) * (xr - xl))
            y_coords.append(yl + (bl / (bl + br + b)) * (yr - yl))
            z_coords.append(bed_level + dbf)
            # point 4
            x_coords.append(xl + (bl / (bl + br + b)) * (xr - xl))
            y_coords.append(yl + (bl / (bl + br + b)) * (yr - yl))
            z_coords.append(bed_level)
            # point 5
            x_coords.append(xl + ((bl + b) / (bl + br + b)) * (xr - xl))
            y_coords.append(yl + ((bl + b) / (bl + br + b)) * (yr - yl))
            z_coords.append(bed_level)
            # point 6
            x_coords.append(xl + ((bl + b) / (bl + br + b)) * (xr - xl))
            y_coords.append(yl + ((bl + b) / (bl + br + b)) * (yr - yl))
            z_coords.append(bed_level + dbf)
            # point 7
            x_coords.append(xr)
            y_coords.append(yr)
            z_coords.append(bed_level + dbf + inter_pr_height)
        else:
            # point 2
            x_coords.append(xl)
            y_coords.append(yl)
            z_coords.append(bed_level + inter_pl_height)
            # point 3
            x_coords.append(xl + (bl / (bl + br + b)) * (xr - xl))
            y_coords.append(yl + (bl / (bl + br + b)) * (yr - yl))
            z_coords.append(bed_level)
            # point 4
            x_coords.append(xl + ((bl + b) / (bl + br + b)) * (xr - xl))
            y_coords.append(yl + ((bl + b) / (bl + br + b)) * (yr - yl))
            z_coords.append(bed_level)
            # point 5
            x_coords.append(xr)
            y_coords.append(yr)
            z_coords.append(bed_level + inter_pr_height)

        # point 8
        x_coords.append(xr)
        y_coords.append(yr)
        z_coords.append(bank_right_level)

        return x_coords, y_coords, z_coords

    @staticmethod
    def get_vortices(
        h: float, hl: float, hr: float, bl: float, br: float, b: float, dbf: float
    ):
        """get_vortices.

            get_vortices calculates the coordinates of all the points (vortices) of the cross-section.

        Parameters
        ----------
        h:[Float]
            water depth
        hl: [Float]
            InterPLHeight Intermediate point right height above the bankfull level.
        hr: [Float]
            InterPRHeight Intermediate point right height above the bankfull level.
        bl: [Float]
            DESCRIPTION.
        br: [Float]
            DESCRIPTION.
        b: [Float]
            DESCRIPTION.
        dbf: [Float/Bool]
            DESCRIPTION.

        Returns
        -------
        x_coords: [List]
            DESCRIPTION.
        y_coords: [List]
            DESCRIPTION.
        z_coords: [List]
            DESCRIPTION.
        """
        # left side slope  Horizontal: Vertical = 1:sl
        sl = hl / bl
        # right side slope Horizontal: Vertical = 1:sr
        sr = hr / br
        # put vertexes to the local coordination system
        x_coords = list()
        y_coords = list()

        if h <= dbf:
            # Point 1
            x_coords.append(0)
            y_coords.append(h)
            # Point 2
            x_coords.append(0)
            y_coords.append(0)
            # Point 3
            x_coords.append(b)
            y_coords.append(0)
            # Point 4
            x_coords.append(b)
            y_coords.append(h)
            # PropXS8P = PolygonGeometry(mp)
        elif h - dbf < min(hl, hr):
            h_new = h - dbf
            # Trapezoidal cross-section with 4 points.
            # case 1
            # Point 1
            x_coords.append(0)
            y_coords.append(h)
            # Point 2
            x_coords.append(h_new / sl)
            y_coords.append(dbf)
            # Point 3
            x_coords.append(h_new / sl)
            y_coords.append(0)
            # Point 4
            x_coords.append((h_new / sl) + b)
            y_coords.append(0)
            # Point 5
            x_coords.append((h_new / sl) + b)
            y_coords.append(dbf)
            # Point 6
            x_coords.append(h_new / sl + b + h_new / sr)
            y_coords.append(h)
        elif h - dbf < max(hl, hr) and hl < hr:
            h_new = h - dbf
            # the height of one of the slopes is higher than the water depth.
            # so the shape of the cross-section is 5 points.
            # case 2
            # Point 1
            x_coords.append(0)
            y_coords.append(h)
            # Point 2
            x_coords.append(0)
            y_coords.append(hl + dbf)
            # Point 3
            x_coords.append(bl)
            y_coords.append(dbf)
            # Point 4
            x_coords.append(bl)
            y_coords.append(0)
            # Point 5
            x_coords.append(bl + b)
            y_coords.append(0)
            # Point 6
            x_coords.append(bl + b)
            y_coords.append(dbf)
            # Point 7
            x_coords.append(bl + b + h_new / sr)
            y_coords.append(h)
        elif h - dbf < max(hl, hr) and hl > hr:
            h_new = h - dbf
            # case 3
            # Point 1
            x_coords.append(0)
            y_coords.append(h)
            # Point 2
            x_coords.append(h_new / sl)
            y_coords.append(dbf)
            # Point 3
            x_coords.append(h_new / sl)
            y_coords.append(0)
            # Point 4
            x_coords.append(h_new / sl + b)
            y_coords.append(0)
            # Point 5
            x_coords.append(h_new / sl + b)
            y_coords.append(dbf)
            # Point 6
            x_coords.append(h_new / sl + b + br)
            y_coords.append(hr + dbf)
            # Point 7
            x_coords.append(h_new / sl + b + br)
            y_coords.append(h)
        else:
            #       elif H - dbf  > max(hl,hr):
            # h_new = H - dbf
            # the whole 6-point cross-section.
            # case 4
            # Point 1
            x_coords.append(0)
            y_coords.append(h)
            # Point 2
            x_coords.append(0)
            y_coords.append(hl + dbf)
            # Point 3
            x_coords.append(bl)
            y_coords.append(dbf)
            # Point 4
            x_coords.append(bl)
            y_coords.append(0)
            # Point 5
            x_coords.append(bl + b)
            y_coords.append(0)
            # Point 6
            x_coords.append(bl + b)
            y_coords.append(dbf)
            # Point 7
            x_coords.append(bl + b + br)
            y_coords.append(hr + dbf)
            # Point 8
            x_coords.append(bl + b + br)
            y_coords.append(h)
            # calculate the area & perimeter of the bankful area
            # area
            # PropXS8P(1) = PropXS8P(1) + (dbf * mw)
            # perimeter
            # PropXS8P(2) = PropXS8P(2) + (2*dbf)
        coords = np.array([x_coords, y_coords]).transpose()

        return coords

    def get_rating_curve(
        self, max_h: float = 20, interval: float = 0.02, dx: float = 500
    ):
        """get_rating_curve.

            get_rating_curve calculates the depth coresponding to each discharge value for a given cross-section using
            the manning equation.

        Parameters
        ----------
        max_h: TYPE, optional
            DESCRIPTION. The default is 20.
        interval: TYPE, optional
            DESCRIPTION. The default is 0.02.
        dx: TYPE, optional
            DESCRIPTION. The default is 500.

        Returns
        -------
        None.
        """
        so = self.slope / dx
        # Rating Curve
        geom = self.cross_sections.loc[self.cross_sections.index[0], :]
        # 0  1   2   3  4 5  6  7  8  9  10 11 12 13 14 15
        "id,xsid,gl,zl,zr,hl,hr,bl,br,xl,yl,xr,yr,b,m,dbf"

        n_int = int(
            (max(geom["zl"] - geom["gl"], geom["zr"] - geom["gl"]) + max_h) / interval
        )

        table = np.zeros(shape=(n_int, 10))
        hq = np.zeros(shape=(n_int, 2))

        for i in range(n_int):
            table[i, 0] = interval * (i + 1)
            # calculate area & perimeter
            coords = self.get_vortices(
                table[i, 0],
                geom["hl"],
                geom["hr"],
                geom["bl"],
                geom["br"],
                geom["b"],
                geom["dbf"],
            )
            table[i, 1:3] = self.polygon_geometry(coords)

            # area & perimeter of the Upper part only
            if table[i, 0] <= geom["dbf"]:
                table[i, 3] = 0
                table[i, 4] = 0
            else:
                # area of upper part only
                table[i, 3] = table[i, 1] - geom["dbf"] * geom["b"]
                # perimeter of upper part
                table[i, 4] = table[i, 2] - 2 * geom["dbf"] - geom["b"]

            # area & perimeter of the lower part only
            table[i, 5] = table[i, 1] - table[i, 3]
            table[i, 6] = table[i, 2] - table[i, 4]

            # Discharge of Upper & lower
            if table[i, 0] <= geom["dbf"]:
                table[i, 7] = 0
            else:
                table[i, 7] = table[i, 3] * ((table[i, 3] / table[i, 4]) ** (2.0 / 3.0))

            table[i, 8] = table[i, 5] * ((table[i, 5] / table[i, 6]) ** (2.0 / 3.0))

            table[i, 7] = (1.0 / geom["m"]) * table[i, 7] * (abs(so) ** 0.5)
            table[i, 8] = (1.0 / geom["m"]) * table[i, 8] * (abs(so) ** 0.5)

            # total discharge
            table[i, 9] = table[i, 7] + table[i, 8]

        hq[:, 0] = table[:, 0]
        hq[:, 1] = table[:, 9]

        self.HQ = hq[:, :]

    def get_days(
        self, from_day: int, to_day: int, delimiter: str = r"\s+", verbose: bool = True
    ):
        r"""get_days.

            get_days method checks if input days exist in the 1D result data or not since hydraulic model simulates only
            days where discharge is above a certain value (2-year return period), you have to enter the
            one_d_result_path attributes of the instance first to read the results of the given sub-basin.

        Parameters
        ----------
        from_day : [int]
            the day you want to read the result from.
        to_day : [int]
            the day you want to read the result to.
        delimiter: [str]
            delimiter used in the 1D result files. Default is r"\s+".
        verbose: [bool]
            to print details of the function.

        Returns
        -------
        Message: [str]
            stating whether the given days exist or not, and if not, two alternatives are given instead.
            (the earliest day before the given day and the earliest day after the given day).
        """
        self._read_1d_results(self.id, delimiter=delimiter, fill_missing=False)
        data = self.results_1d
        days = list(set(data["day"]))
        days.sort()

        if from_day not in days:
            alternative_1 = from_day
            stop = 0
            # search for from_day in the days column.
            while stop == 0:
                try:
                    np.where(data["day"] == alternative_1)[0][0]
                    stop = 1
                except IndexError:
                    alternative_1 = alternative_1 - 1
                    # logger.debug(alternative_1)
                    if alternative_1 <= 0:
                        stop = 1
                    continue

            alternative_2 = from_day
            # search for the closest later days
            stop = 0
            while stop == 0:
                try:
                    np.where(data["day"] == alternative_2)[0][0]
                    stop = 1
                except IndexError:
                    alternative_2 = alternative_2 + 1
                    if alternative_2 >= data.loc[len(data) - 1, "day"]:
                        stop = 1
                    continue
            if verbose:
                text = f"""
                the from_day you entered does not exist in the data, and the closest day earlier than your input day is  {alternative_1} and the closest later day is {alternative_2}
                """
                logger.debug(text)
            if abs(alternative_1 - from_day) > abs(alternative_2 - from_day):
                alternative_1 = alternative_2
        else:
            if verbose:
                logger.debug("from_day you entered does exist in the data ")
            alternative_1 = from_day

        # if to_day does not exist in the results
        if to_day not in days:
            alternative_3 = to_day
            stop = 0
            # search for the to_day in the days column
            while stop == 0:
                try:
                    np.where(data["day"] == alternative_3)[0][0]
                    stop = 1
                except IndexError:
                    alternative_3 = alternative_3 - 1
                    if alternative_3 <= 0:
                        stop = 1
                    continue

            alternative_4 = to_day
            # search for the closest later days
            stop = 0
            while stop == 0:
                try:
                    np.where(data["day"] == alternative_4)[0][0]
                    stop = 1
                except IndexError:
                    alternative_4 = alternative_4 + 1
                    if alternative_4 >= data.loc[len(data) - 1, "day"]:
                        alternative_4 = data.loc[len(data) - 1, "day"]
                        stop = 1
                    continue
            if verbose:
                text = f"""
                the to_day you entered does not exist in the data, and the closest day earlier than your input day is  {alternative_3} and the closest later day is {alternative_4}
                """
                logger.debug(text)

            if abs(alternative_3 - to_day) > abs(alternative_4 - to_day):
                alternative_3 = alternative_4
        else:
            if verbose:
                logger.debug("to_day you entered does exist in the data ")

            alternative_3 = to_day

        return alternative_1, alternative_3

    # @staticmethod
    # def correct_maps(DEMpath, Filelist, Resultpath, MapsPrefix, FilterValue, Saveto):
    #     """CorrectMaps.
    #
    #     CorrectMaps method check every 2D result that starts with the given Mapsprefix
    #     and replace the Nan value with zeros and the values higher than 99 with zeros
    #
    #     Parameters
    #     ----------
    #     DEMpath : [String]
    #         path to the DEM ascii file including the name and extension
    #         (i.e., c/files/RhineDEM.asc) .
    #     Filelist : [String]
    #         - if you have a list of files to correct enter the Filelist as the path to the file
    #         containing the names
    #         ex,
    #             Filelist = "F:/RFM/RIM_all/RIM1.0/M35(done)/errorlist.txt"
    #
    #         - if you want to check all the files in the resultpath enter the
    #         Filelist as '0'
    #         ex,
    #             Filelist = '0'
    #     Resultpath : [String]
    #         path where the Maps exist.
    #     MapsPrefix : [String]
    #         the name prefix that distinguish the maps you want to correct from
    #         other maps in the same folder, like the first part of the name you
    #         use to name all files.
    #     FilterValue: []
    #
    #     Saveto : [String]
    #         path to where you will save the corrected files.
    #
    #     Returns
    #     -------
    #     Errors : [list]
    #         list of the files' names that has errors and are already corrected.
    #     """
    #     DEM, SpatialRef = raster.readASCII(DEMpath)
    #     NoDataValue = SpatialRef[-1]
    #
    #     # filter and get the required maps
    #     if Filelist == "0":
    #         # read list of file names
    #         AllResults = os.listdir(Resultpath)
    #
    #         MapsNameList = list()
    #         for i in range(len(AllResults)):
    #             if AllResults[i].startswith(MapsPrefix):
    #                 MapsNameList.append(AllResults[i])
    #     elif type(Filelist) == str:
    #         MapsNameList = pd.read_csv(Filelist, header=None)[0].tolist()
    #
    #     Errors = list()
    #
    #     for k in range(len(MapsNameList)):
    #         try:
    #             # open the zip file
    #             compressedfile = zipfile.ZipFile(Resultpath + "/" + MapsNameList[k])
    #         except:
    #             logger.debug("Error Opening the compressed file")
    #             Errors.append(MapsNameList[k][len(MapsPrefix) : -4])
    #             continue
    #
    #         # get the file name
    #         fname = compressedfile.infolist()[0]
    #         # get the time step from the file name
    #         timestep = int(fname.filename[len(MapsPrefix) : -4])
    #         logger.debug("File No = " + str(k))
    #
    #         ASCIIF = compressedfile.open(fname)
    #         SpatialRef = ASCIIF.readlines()[:6]
    #         ASCIIF = compressedfile.open(fname)
    #         ASCIIRaw = ASCIIF.readlines()[6:]
    #         rows = len(ASCIIRaw)
    #         cols = len(ASCIIRaw[0].split())
    #         MapArray = np.ones((rows, cols), dtype=np.float32) * 0
    #         # read the ascii file
    #         for i in range(rows):
    #             x = ASCIIRaw[i].split()
    #             MapArray[i, :] = list(map(float, x))
    #
    #         Save = 0
    #         # Clip all maps
    #         if MapArray[DEM == NoDataValue].max() > 0:
    #             MapArray[DEM == NoDataValue] = 0
    #             Save = 1
    #         # replace nan values with zero
    #         if len(MapArray[np.isnan(MapArray)]) > 0:
    #             MapArray[np.isnan(MapArray)] = 0
    #             Save = 1
    #         # replace FilterValue in the result raster with 0
    #         if len(MapArray[MapArray >= FilterValue]) > 0:
    #             MapArray[MapArray >= FilterValue] = 0
    #             Save = 1
    #
    #         if Save == 1:
    #             logger.debug("File= " + str(timestep))
    #             # write the new file
    #             fname = MapsPrefix + str(timestep) + ".asc"
    #             newfile = Saveto + "/" + fname
    #
    #             with open(newfile, "w") as File:
    #                 # write the first lines
    #                 for i in range(len(SpatialRef)):
    #                     File.write(str(SpatialRef[i].decode()[:-2]) + "\n")
    #
    #                 for i in range(rows):
    #                     File.writelines(list(map(raster.stringSpace, MapArray[i, :])))
    #                     File.write("\n")
    #
    #             # zip the file
    #             with zipfile.ZipFile(
    #                 Saveto + "/" + fname[:-4] + ".zip", "w", zipfile.ZIP_DEFLATED
    #             ) as newzip:
    #                 newzip.write(Saveto + "/" + fname, arcname=fname)
    #             # delete the file
    #             os.remove(Saveto + "/" + fname)
    #
    #     return Errors
    #
    # def listAttributes(self):
    #     """ListAttributes.
    #
    #     Print Attributes List
    #     """
    #     logger.debug("\n")
    #     logger.debug(
    #         f"Attributes List of: {repr(self.__dict__['name'])} - {self.__class__.__name__} Instance\n"
    #     )
    #     self_keys = list(self.__dict__.keys())
    #     self_keys.sort()
    #     for key in self_keys:
    #         if key != "name":
    #             logger.debug(str(key) + " : " + repr(self.__dict__[key]))
    #
    #     logger.debug("\n")


class Reach(River):
    """Reach.

    represent a segment of the river to create the Reach instance the
    river object has to have the cross-sections read using the
    'ReadCrossSections' method
    """

    xs_hydrograph: DataFrame

    reach_attr = dict(
        extracted_values=dict(),
        xs_hydrograph=None,
        neg_qmin=None,
        Negative=None,
        xs_water_level=None,
        xs_water_depth=None,
        RRM=None,
        RRM2=None,
        resampled_q=None,
        resampled_wl=None,
        resampled_h=None,
        Qrp=None,
        detailed_overtopping_left=None,
        detailed_overtopping_right=None,
        all_overtopping_vs_xs=None,
        all_overtopping_vs_time=None,
        BC=None,
        area_per_high=None,
        area_per_Low=None,
        total_flow=None,
        rrm_progression=None,
        laterals_table=None,
        Laterals=None,
        results_1d=None,
        us_hydrographs=None,
        last_reach=False,
    )

    @class_attr_initialize(reach_attr)
    def __init__(
        self, sub_id: int, entire_river: River, run_model: bool = False, *args, **kwargs
    ):
        # initialize the attributes with the river attributes
        for key, val in entire_river.__dict__.items():
            setattr(self, key, val)

        self.id = sub_id
        # if the river reach is the last reach in the river
        if sub_id == entire_river.reach_ids[-1]:
            self.last_reach = True

        if not isinstance(entire_river.cross_sections, DataFrame):
            raise ValueError(
                "please Read the cross-section for the whole river with 'ReadCrossSections' "
                "method before creating the Reach instance"
            )
        # filter the whole cross-section file and get the cross-section of the segment
        self.cross_sections = entire_river.cross_sections[
            entire_river.cross_sections["id"] == sub_id
        ]
        self._getXS(run_model=run_model)

        if (
            isinstance(entire_river.slope, DataFrame)
            and self.id in entire_river.slope["id"].tolist()
        ):
            self.slope = entire_river.slope[entire_river.slope["id"] == sub_id][
                "slope"
            ].tolist()[0]

        if isinstance(entire_river.river_network, DataFrame):
            self.upstream_node, self.downstream_node = entire_river.trace_segment(
                sub_id
            )
        else:
            self.upstream_node, self.downstream_node = [], []

        if isinstance(entire_river.RP, DataFrame):
            self.RP = entire_river.RP.loc[
                entire_river.RP["node"] == self.upstream_node, ["HQ2", "HQ10", "HQ100"]
            ]

        if isinstance(entire_river.SP, DataFrame):
            self.SP = entire_river.SP.loc[entire_river.SP["id"] == self.id, :]
            self.SP.index = list(range(len(self.SP)))

    def _getXS(self, run_model: bool):
        """Get the cross-sections of the current river reach.

        Parameters
        ----------
        run_model: [bool]
            If True the values (as array) for each attribute of the cross-section will be stored in the reach object.

        Returns
        -------
        cross_sections: [DataFrame]
            Replaces the cross_sections attributes in the reach object from the whole river cross-sections
            to the cross-section of the current reach only.
        last_xs: [int]
            the id of the last cross-section.
        first_xs: [int]
            the id of the last cross-section.
        xs_names: [List]
            list of current reach cross-sections id.
        xs_number: [int]
            number of cross-sections in the current river reach.
        """
        if run_model:
            self.xsid = self.cross_sections.loc[:, "xsid"].values
            self.dbf = self.cross_sections.loc[:, "dbf"].values
            self.bed_level = self.cross_sections.loc[:, "gl"].values
            self.hl = self.cross_sections.loc[:, "hl"].values
            self.cl = self.cross_sections.loc[:, "bl"].values
            self.zl = self.cross_sections.loc[:, "zl"].values
            self.hr = self.cross_sections.loc[:, "hr"].values
            self.cr = self.cross_sections.loc[:, "br"].values
            self.zr = self.cross_sections.loc[:, "zr"].values
            self.mw = self.cross_sections.loc[:, "b"].values
            self.mn = self.cross_sections.loc[:, "m"].values

        self.cross_sections.index = list(range(len(self.cross_sections)))
        self.last_xs = self.cross_sections.loc[len(self.cross_sections) - 1, "xsid"]
        self.first_xs = self.cross_sections.loc[0, "xsid"]
        self.xs_names = self.cross_sections["xsid"].tolist()
        self.xs_number = len(self.xs_names)

    def extract_results(self, xs_id: int, variable: str = "q") -> pd.Series:
        """Extract XS results.

            - Extract the results from the 1D dataframe (results_1d) for a given xs and convert it from the form of
            day, hour, <q>, to datetime, <q>, and merge it to the xs_hydrograph, xs_water_level, xs_water_depth

        Parameters
        ----------
        xs_id: [int]
            cross-section id.
        variable: [str]
            h, q, wl

        Returns
        -------
        pd.Series
        """
        # first extract the xs results
        f = self.results_1d.loc[
            self.results_1d["xs"] == xs_id, ["day", "hour", variable]
        ].reset_index(drop=True)
        # get the gregorian date from the ordinal date.
        f["date"] = f.apply(lambda x: self._get_date(x["day"], x["hour"]), axis=1)
        if variable == "q":
            g = self.xs_hydrograph.merge(
                f, how="left", left_index=True, right_on="date"
            )[variable].values
        elif variable == "h":
            g = self.xs_water_depth.merge(
                f, how="left", left_index=True, right_on="date"
            )[variable].values
        elif variable == "wl":
            g = self.xs_water_level.merge(
                f, how="left", left_index=True, right_on="date"
            )[variable].values
        else:
            raise ValueError("variable should be either 'q', 'h', or 'wl'")

        # data = self.results_1d
        # g = data.loc[data["xs"] == xs_id, :]
        # g.drop_duplicates(inplace=True)
        return g

    def read_1d_results(
        self,
        from_day: Union[int, str] = None,
        to_day: Union[int, str] = None,
        fill_missing: bool = True,
        add_hq2: bool = False,
        path: str = None,
        xsid: int = None,
        chunk_size: int = None,
        delimiter: str = r"\s+",
        extension: str = ".txt",
    ):
        r"""read1DResult.

        - Read1DResult method reads the 1D result of the river reach the method is returns the hydrograph of the first
        and last cross-section.
        - the method will not read the 1D result file again if you tried to read results of the same sub-basin again,
        so you have to re-instantiate the object.

        Parameters
        ----------
        from_day: [integer], optional
            the order of the day you want the data to start from.
            The default is None. it means read everything
        to_day: [integer], optional
            the order of the day you want the data to end to. The default
            is empty. means read everything
        fill_missing: [Bool], optional
            Fill the missing days with zeroes. The default is True.
        add_hq2: [Bool], optional
            to add the value of HQ2. The default is False.
        path: [String], optional
            path to read the results from. The default is None.
        xsid: [Integer], optional
            id of a specific cross-section you want to extract the results for
            it. The default is None.
        chunk_size: [int]
            size of the chunk if you want to read the file in chunks Default is = None.
        delimiter: [str]
            delimiter separating the columns in the result file. Default is r"\s+", which is a space delimiter.
        extension: [str]
            The extension of the file. Default is ".txt"

        Returns
        -------
        results_1d : [attribute]
            The results read from the file as is, will be stored in the attribute "results_1d".
        xs_hydrograph : [dataframe attribute]
            dataframe containing hydrographs at the position of the first.
            and last cross-section
        xs_water_level : [dataframe attribute]
            dataframe containing water levels at the position of the first.
            and last cross-section
        first_day_results:[attribute]
            the first day in the 1D result.
        last_day:[attribute]
            the last day in the 1D result.
        """
        if path is None and self.one_d_result_path is None:
            raise ValueError(
                "User have to either enter the value of the 'path' parameter or"
                " define the 'one_d_result_path' parameter for the River object"
            )
        # if the results are not read yet read it
        if not isinstance(self.results_1d, DataFrame):
            self._read_1d_results(
                self.id,
                from_day,
                to_day,
                path=path,
                fill_missing=fill_missing,
                chunk_size=chunk_size,
                delimiter=delimiter,
                extension=extension,
            )
        # get the index of the days and convert them into dates.
        if not from_day:
            from_day = self.results_1d.loc[0, "day"]
        if not to_day:
            to_day = self.results_1d.loc[len(self.results_1d) - 1, "day"]

        start = self.ordinal_to_date(from_day)
        end = self.ordinal_to_date(to_day + 1)

        if not isinstance(self.xs_hydrograph, DataFrame):
            self.xs_hydrograph = pd.DataFrame(
                index=pd.date_range(start, end, freq="H")[:-1]
            )
            self.xs_water_level = pd.DataFrame(
                index=pd.date_range(start, end, freq="H")[:-1]
            )
            self.xs_water_depth = pd.DataFrame(
                index=pd.date_range(start, end, freq="H")[:-1]
            )

        # check if the xsid is in the sub-basin.
        if xsid:
            if xsid not in self.xs_names:
                raise ValueError(
                    f"The given cross-section {xsid} does not exist inside the "
                    f"current Segment of the river, first XS is {self.first_xs}, and last "
                    f"XS is {self.last_xs}"
                )

        # get the simulated hydrograph and add the cut HQ2.
        if add_hq2:
            self.xs_hydrograph[self.last_xs] = (
                self.results_1d.loc[self.results_1d["xs"] == self.last_xs, "q"].values
                + self.RP["HQ2"].tolist()[0]
            )
            self.xs_hydrograph[self.first_xs] = (
                self.results_1d.loc[self.results_1d["xs"] == self.first_xs, "q"].values
                + self.RP["HQ2"].tolist()[0]
            )

            if xsid:
                self.xs_hydrograph[xsid] = (
                    self.results_1d.loc[self.results_1d["xs"] == xsid, "q"].values
                    + self.RP["HQ2"].tolist()[0]
                )
        else:
            self.xs_hydrograph[self.last_xs] = self.extract_results(
                self.last_xs, variable="q"
            )
            self.xs_hydrograph[self.first_xs] = self.extract_results(
                self.first_xs, variable="q"
            )
            if xsid:
                self.xs_hydrograph[xsid] = self.extract_results(xsid, variable="q")

        self.xs_water_level[self.last_xs] = self.extract_results(
            self.last_xs, variable="wl"
        )
        self.xs_water_level[self.first_xs] = self.extract_results(
            self.first_xs, variable="wl"
        )
        self.xs_water_depth[self.last_xs] = self.extract_results(
            self.last_xs, variable="h"
        )
        self.xs_water_depth[self.first_xs] = self.extract_results(
            self.first_xs, variable="h"
        )

        if xsid:
            self.xs_water_level[xsid] = self.extract_results(xsid, variable="wl")
            self.xs_water_depth[xsid] = self.extract_results(xsid, variable="h")

        self.xs_water_level.dropna(axis=0, inplace=True)
        self.xs_water_depth.dropna(axis=0, inplace=True)
        self.xs_hydrograph.dropna(axis=0, inplace=True)
        # check the first day in the results and get the date of the first day and last day
        # create time series
        self.from_beginning = self.results_1d["day"][0]
        self.first_day = self.ordinal_to_date(self.from_beginning)
        # if there are empty days at the beginning the filling missing days is
        # not going to detect it so ignore it here by starting from the first
        # day in the data (data['day'][0]) dataframe empty days at the
        # beginning
        # TODO
        # the from_beginning and first_day_results are exactly the same
        # delete one of them
        self.first_day_results = self.ordinal_to_date(self.results_1d.loc[0, "day"])
        last_day = self.results_1d.loc[self.results_1d.index[-1], "day"]
        self.last_day = self.ordinal_to_date(last_day)

        # last days+1 as range does not include the last element
        self.days_list = list(
            range(
                self.results_1d.loc[0, "day"],
                self.results_1d.loc[self.results_1d.index[-1], "day"] + 1,
            )
        )
        self.reference_index_results = pd.date_range(
            self.first_day_results, self.last_day, freq="D"
        )

    def extract_xs(self, xsid: int, add_hq2: bool = False, water_level: bool = True):
        """extract_xs.

            ExtractXS method extracts the hydrodraph and water levels of a specific
            xsid from the already been read 1D results

        Parameters
        ----------
        xsid : [Integer], optional
            id of a specific cross-section you want to get the results on it.
        add_hq2 : [Bool], optional
            to add the value of HQ2. The default is False.

        Returns
        -------
        None.
        """
        assert isinstance(
            self.results_1d, DataFrame
        ), "please use the Read1DResult method to read the results first"
        # assert hasattr(self,"RP"), "please use the Read1DResult method to read the results first"
        if add_hq2:
            self.xs_hydrograph[xsid] = (
                self.results_1d["q"][self.results_1d["xs"] == xsid].values
                + self.RP["HQ2"].tolist()[0]
            )
        else:
            self.xs_hydrograph[xsid] = self.results_1d["q"][
                self.results_1d["xs"] == xsid
            ].values

        if water_level:
            self.xs_water_level[xsid] = self.results_1d["wl"][
                self.results_1d["xs"] == xsid
            ].values

    def check_negative_q(self, plot: bool = False, temporal_resolution: str = "hourly"):
        """check_negative_q.

            check_negative_q check whether there are any negative discharge values in the 'q' column in the 1D
            results or not, you need to read the result first.

        Returns
        -------
        Negative.[attribute]
            dictionary with ['NegQ', 'NegXS', 'NegQind'] as keys.
        """
        if temporal_resolution == "hourly":
            assert isinstance(
                self.results_1d, DataFrame
            ), "please use the results_1d method to read the result of this sub-basin first"

            if self.results_1d["q"].min() < 0:
                logger.debug("NegativeDischarge")
                # extract -ve discharge data if exist
                self.Negative = dict()
                self.Negative["NegQ"] = self.results_1d[self.results_1d["q"] < 0]
                self.Negative["NegXS"] = list(set(self.Negative["NegQ"]["xs"]))
                self.Negative["NegQind"] = self.Negative["NegQ"].index.tolist()

                self.Negative["QN"] = pd.DataFrame()
                for i in range(len(self.Negative["NegXS"])):
                    self.Negative["QN"][self.Negative["NegXS"][i]] = self.results_1d[
                        "q"
                    ][self.results_1d["xs"] == self.Negative["NegXS"][i]]

                self.Negative["QN"].index = self.xs_hydrograph.index

                if plot:
                    plt.figure(30, figsize=(15, 8))
                    for i in range(len(self.Negative["NegXS"])):
                        plt.plot(self.Negative["QN"][self.Negative["NegXS"][i]])

                    plt.title("Discharge ", fontsize=25)
                    plt.legend(self.Negative["NegXS"], fontsize=15)
                    plt.xlabel("Time", fontsize=15)
                    plt.ylabel("Discharge m3/s", fontsize=15)

            else:
                logger.debug("There is no -ve Discharge")

        elif temporal_resolution == "1min":
            assert hasattr(
                self, "q"
            ), "please use the results_1d method to read the result of this sub-basin first"
            # neg_qmin = pd.DataFrame()
            NegQmin = self.q
            NegQmin.loc[:, "date"] = self.q.index[:]
            NegQmin.index = range(len(NegQmin.index))
            f = NegQmin[NegQmin[self.xs_names[0]] < 0]

            for i in range(len(self.xs_names[1:])):
                f = f.append(NegQmin[NegQmin[self.xs_names[i + 1]] < 0])

            self.NegQmin = f

    def read_rrm_hydrograph(
        self,
        station_id: int,
        from_day: Union[int, str] = None,
        to_day: Union[int, str] = None,
        path: str = None,
        date_format: str = "%d_%m_%Y",
        location: int = 1,
        path2: str = None,
    ):
        """read_rrm_hydrograph.

            read_rrm_hydrograph method reads the results of the Rainfall-runoff model for the given node id for
            a specific period.

        Parameters
        ----------
        station_id : [Integer]
            DESCRIPTION.
        from_day: [Integer], optional
            start day of the period you want to read its results. The default is [].
        to_day: [Integer], optional
            end day of the period you want to read its results.
            The default is [].
        path: [str]
            path to the directory where the result files. if not given, the river.rrm_path should be given. default
            is ''
        date_format: [str]
            format of the date string, default is "%d_%m_%Y"
        location: [1]
            1 if there are results for the rainfall run of model at one location, 2 if the are results for the rrm at
            two locations. default is 1.
        path2: [str]
            path where the results of the rrm at the second location, this parameter is needed only in case
            location=2. default is ''

        Returns
        -------
        RRM:[data frame attribute]
            containing the computational node and rainfall-runoff results
            (hydrograph)with columns ['id', Nodeid ]
        """
        if not isinstance(self.RRM, DataFrame):
            self.RRM = pd.DataFrame()

        if location == 2 and not isinstance(self.RRM2, DataFrame):
            # create a dataframe for the 2nd time series of the rainfall runoff
            # model at the second location
            self.RRM2 = pd.DataFrame()

        if not path:
            path = self.rrm_path

        if location == 2 and not path2:
            raise ValueError(
                "path2 argument has to be given for the location of the 2nd rainfall run-off time series"
            )

        if location == 1:
            self.RRM[station_id] = self._read_rrm_results(
                self.version,
                self.rrm_reference_index,
                path,
                station_id,
                from_day,
                to_day,
                date_format,
            )[station_id].tolist()
        else:
            self.RRM[station_id] = self._read_rrm_results(
                self.version,
                self.rrm_reference_index,
                path,
                station_id,
                from_day,
                to_day,
                date_format,
            )[station_id].tolist()
            try:
                self.RRM2[station_id] = self._read_rrm_results(
                    self.version,
                    self.rrm_reference_index,
                    path2,
                    station_id,
                    from_day,
                    to_day,
                    date_format,
                )[station_id].tolist()
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"The directory you have given for the location 2 {path2}, is not correct "
                    f"please check"
                )

        logger.info("RRM time series for the gauge " + str(station_id) + " is read")

        if not from_day:
            from_day = 1
        if not to_day:
            to_day = len(self.RRM[station_id])

        start = self.rrm_reference_index.loc[from_day, "date"]
        end = self.rrm_reference_index.loc[to_day, "date"]

        if location == 1:
            self.RRM.index = pd.date_range(start, end, freq="D")
        else:
            self.RRM.index = pd.date_range(start, end, freq="D")
            self.RRM2.index = pd.date_range(start, end, freq="D")
        # get the simulated hydrograph and add the cut HQ2

    def resample(
        self,
        xsid,
        column_name,
        from_day: Union[int, str] = "",
        to_day: Union[int, str] = "",
        delete: bool = False,
    ):
        """resample.

            resample method extracts the value at the last hour of the dat.

            [hour == 24] from the 1D Result file, for the discharge, water level, and water depth.

        Parameters
        ----------
        xsid: [Integer]
            cross-section id.
        column_name: [string]
            the column name you want to resample in the results1D. column_name could be 'q' for discharge,
            'wl' for water level, and 'h' for water depth.
        from_day: [integer], optional
            starting day. The default is ''.
        to_day: [integer], optional
            end day. The default is ''.
        delete: [boolen], optional
            to delete the previously resampled data frame to create another one. The default is False.

        Returns
        -------
        resampled_q, resampled_wl, resampled_h: [dataframe attribute]
            depends on the given column_name the attribute will be created, if 'q' the attribute will be resampled_q, and the same for "wl", and "H"
            and inside the resampled_q a column will be created with the given xsid containing the resampled values.
        """
        assert hasattr(self, "results_1d"), "please read the 1D results"

        if from_day == "":
            from_day = self.results_1d.loc[0, "day"]
        if to_day == "":
            to_day = self.results_1d.loc[len(self.results_1d) - 1, "day"]

        # start = self.IndexToDate(from_day)
        # end = self.IndexToDate(to_day)

        # start = self.reference_index.loc[from_day,'date']
        # end = self.reference_index.loc[to_day,'date']

        ind = pd.date_range(
            self.ordinal_to_date(from_day), self.ordinal_to_date(to_day), freq="D"
        )

        if column_name == "q" and not hasattr(self, "resampled_q"):
            self.resampled_q = pd.DataFrame(index=ind)
        elif column_name == "q":
            if delete:
                del self.resampled_q

        if column_name == "wl" and not hasattr(self, "resampled_wl"):
            self.resampled_wl = pd.DataFrame(index=ind)
        elif column_name == "wl":
            if delete:
                del self.resampled_wl

        if column_name == "h" and not hasattr(self, "resampled_h"):
            self.resampled_h = pd.DataFrame(index=ind)
        elif column_name == "h":
            if delete:
                del self.resampled_h

        q = self.results_1d[self.results_1d["xs"] == xsid][
            self.results_1d["hour"] == 24
        ]
        q = q[column_name][self.results_1d["day"] >= from_day][
            self.results_1d["day"] <= to_day
        ]

        # self.q = q
        if column_name == "q":
            self.resampled_q.loc[:, xsid] = q.tolist()
        elif column_name == "wl":
            self.resampled_wl.loc[:, xsid] = q.tolist()
        elif column_name == "h":
            self.resampled_h.loc[:, xsid] = q.tolist()

    def detailed_statistical_calculation(
        self, return_period: Union[list, np.ndarray], distribution: str = "GEV"
    ):
        """detailed_statistical_calculation.

            detailed_statistical_calculation method calculates the discharge related to a specific given return period.

        Parameters
        ----------
        return_period : TYPE
            DESCRIPTION.
        distribution: [str]
            distribution name, ["GEV", "Gumbel", "Exponential", "Normal"]. Default is "GEV"

        Returns
        -------
        None.
        """
        assert hasattr(self, "SP"), "you "
        return_period = np.array(return_period)
        cdf = 1 - (1 / return_period)
        self.Qrp = pd.DataFrame()
        self.Qrp["RP"] = return_period

        dist = Distributions(distribution)
        parameters = dict(
            loc=self.SP.loc[0, "loc"],
            scale=self.SP.loc[0, "scale"],
        )
        if distribution == "GEV":
            parameters["shape"] = self.SP.loc[0, "shape"]

        self.Qrp["Q"] = dist.theorical_quantile(parameters, cdf)

    def _get_reach_overtopping(
        self, left: bool, event_days: List[int], delimiter: str = r"\s+"
    ):
        xs_s = self.cross_sections.loc[:, "xsid"].tolist()
        columns = [f"reach-{self.id}"] + xs_s + ["sum"]
        df = pd.DataFrame(index=event_days + ["sum"], columns=columns)
        df.loc[:, columns] = 0

        if left:
            path = rf"{self.one_d_result_path}/{self.id}{self.left_overtopping_suffix}"
        else:
            path = rf"{self.one_d_result_path}/{self.id}{self.right_overtopping_suffix}"

        try:
            data = pd.read_csv(path, header=None, delimiter=delimiter)
        except FileNotFoundError:
            return df

        data.columns = ["day", "hour", "xsid", "q", "wl"]
        # get the days in the sub
        days = set(data.loc[:, "day"])
        for day_j in event_days:
            # check whether this sub-basin has flooded in this particular day.
            if day_j in days:
                # filter the dataframe to the discharge column (3) and the days
                df.loc[day_j, f"reach-{self.id}"] = data.loc[
                    data["day"] == day_j, "q"
                ].sum()
                # get the xss that was overtopped in that particular day
                xs_day = set(data.loc[data["day"] == day_j, "xsid"].tolist())
                for xs_i in xs_day:
                    df.loc[day_j, xs_i] = data.loc[data["day"] == day_j, "q"][
                        data["xsid"] == xs_i
                    ].sum()
            else:
                df.loc[day_j, f"reach-{self.id}"] = 0

            # calculate sum
            df.loc[day_j, "sum"] = df.loc[day_j, xs_s].sum()

        for xs_j in xs_s:
            df.loc["sum", xs_j] = df.loc[:, xs_j].sum()

        df.loc["sum", f"reach-{self.id}"] = df.loc[:, f"reach-{self.id}"].sum()

        return df

    def detailed_overtopping(self, event_days: List[int], delimiter: str = r"\s+"):
        r"""detailed_overtopping.

            - detailed_overtopping method takes a list of days and gets the left and right overtopping for the river
            reach each day.

        Parameters
        ----------
        event_days : [list]
            list of days of an event.
        delimiter: str
            Delimiter used in the 1D result files. Default is r"\s+".

        Returns
        -------
        detailed_overtopping_left:[DataFrame]
            - DataFrame containing the overtopping for each cross-section for each day (unit is m3/s/hour).
            - The value represents the sum of all hourly values (each is m3/s) for all hours that has overtopping
            so to convert it to volume multiply it by 60*60.
                reach-5 16705 16706 16707 16708 16709  ... 16850 16851 16852 16853 16854    sum
            35     75.5     0     0     0     0     0  ...     0     0     0     0     0   75.5
            36    374.3     0     0     0     0     0  ...     0     0     0     0     0  374.3
            37    179.0     0     0     0     0     0  ...     0     0     0     0     0  179.0
            38     15.3     0     0     0     0     0  ...     0     0     0     0     0   15.3
            39        0     0     0     0     0     0  ...     0     0     0     0     0      0
            40        0     0     0     0     0     0  ...     0     0     0     0     0      0
            sum   644.1     0     0     0     0     0  ...     0     0     0     0     0      0
        detailed_overtopping_right:[DataFrame]
            - DataFrame containing the overtopping for each cross-section for each day (unit is m3/s/hour)
            - The value represents the sum of all hourly values (each is m3/s) for all hours that has overtopping
            so to convert it to volume multiply it by 60*60.
                reach-5 16705 16706 16707 16708 16709  ... 16850 16851 16852 16853 16854    sum
            35     95.6     0     0     0     0     0  ...     0     0     0     0     0   95.6
            36    805.6     0     0     0     0     0  ...     0     0     0     0     0  805.6
            37    267.5     0     0     0     0     0  ...     0     0     0     0     0  267.5
            38      9.0     0     0     0     0     0  ...     0     0     0     0     0    9.0
            39        0     0     0     0     0     0  ...     0     0     0     0     0      0
            40        0     0     0     0     0     0  ...     0     0     0     0     0      0
            sum  1177.7     0     0     0     0     0  ...     0     0     0     0     0      0
        all_overtopping_vs_xs:
            - Sum of the overtopping from every cross-section over the whole event.
            - The value represents the sum of all hourly values (each is m3/s) for all hours that has overtopping so
            to convert it to volume multiply it by 60*60.
                      sum
            16760    58.7
            16781     0.7
            16797     7.9
            16806    98.6
            16808   105.3
            16809    64.0
            16810  1166.2
            16833   320.4
        all_overtopping_vs_time:
            - Sum of the overtopping from all cross-sections for each day of the event.
            - The value represents the sum of all hourly values (each is m3/s) for all hours that has overtopping
            so to convert it to volume multiply it by 60*60.
           id  Overtopping       date
            0  35        171.1 1955-02-04
            1  36       1179.9 1955-02-05
            2  37        446.5 1955-02-06
            3  38         24.3 1955-02-07
            4  39          0.0 1955-02-08
            5  40          0.0 1955-02-09
        """
        xs_s = self.cross_sections.loc[:, "xsid"].tolist()
        # Left Bank
        df_left = self._get_reach_overtopping(True, event_days, delimiter=delimiter)
        # right Bank
        df_right = self._get_reach_overtopping(False, event_days, delimiter=delimiter)

        df = (df_left.loc["sum", xs_s] + df_right.loc["sum", xs_s]).to_frame()
        self.all_overtopping_vs_xs = df[df["sum"] > 0]

        self.all_overtopping_vs_time = pd.DataFrame()
        self.all_overtopping_vs_time["id"] = event_days
        self.all_overtopping_vs_time.loc[:, "Overtopping"] = (
            df_left.loc[event_days, "sum"] + df_right.loc[event_days, "sum"]
        ).tolist()

        self.all_overtopping_vs_time.loc[:, "date"] = (
            self.reference_index.loc[event_days[0] : event_days[-1], "date"]
        ).tolist()

        self.detailed_overtopping_right = df_right
        self.detailed_overtopping_left = df_left

    def save_hydrograph(self, xsid: int, path: str = None, option: int = 1):
        """Save Hydrograph.

            - SaveHydrograph method saves the hydrograph of any cross-section in the segment.
            - Mainly, the method is created to be used to save the last cross-section hydrograph to use it as a
            boundary condition for the downstream segment.

        Parameters
        ----------
        xsid: [integer]
            the id of the cross-section.
        path: [string], optional
            path to the directory where you want to save the file to. if not given, the files are going to be saved
            in the 'customized_runs_path' path, The default is ''.
        option : [integer]
            1 to write water depth results, 2 to write water level results.

        Returns
        -------
        None.
        """
        if not path:
            if not self.customized_runs_path:
                raise ValueError(
                    "please enter the value of the customized_runs_path or use the path "
                    "argument to specify where to save the file"
                )
            path = self.customized_runs_path

        ts = self.xs_hydrograph[xsid].resample("D").last().to_frame()
        val = [self.xs_hydrograph[xsid][0]] + self.xs_hydrograph[xsid].resample(
            "D"
        ).last().values.tolist()[:-1]
        ts[xsid] = val

        f = pd.DataFrame(index=ts.index)
        f["date"] = ["'" + str(i)[:10] + "'" for i in ts.index]
        f["discharge(m3/s)"] = ts

        if option == 1:
            val = [self.xs_water_depth[xsid][0]] + self.xs_water_depth[xsid].resample(
                "D"
            ).last().values.tolist()[:-1]
            f["water depth(m)"] = val
        else:
            val = [self.xs_water_level[xsid][0]] + self.xs_water_level[xsid].resample(
                "D"
            ).last().values.tolist()[:-1]
            f["water level(m)"] = val

        f.to_csv(f"{path}{self.id}.txt", index=False, float_format="%.3f")

    def plot_hydrograph_progression(
        self,
        xss: list,
        start: str = None,
        end: str = None,
        from_xs: int = None,
        to_xs: int = None,
        line_width: int = 4,
        spacing: int = 5,
        fig_size: tuple = (7, 5),
        x_labels: Union[bool, int] = False,
        fmt="%Y-%m-%d",
    ) -> Tuple[Figure, object]:
        """plot_hydrograph_progression.

            - plot the hydrograph for several cross-section in the segment, cross-section are chosen based on the
            spacing (spacing equal 5 means from the beginning take every fifth cross-section).

        Parameters
        ----------
        xss: [list]
            DESCRIPTION.
        start: TYPE
            DESCRIPTION.
        end: TYPE
            DESCRIPTION.
        from_xs: [str, int]
            default "".
        to_xs: [str, int]
            default is ""
        line_width: [integer], optional
            width of the plots. The default is 4.
        spacing: [integer]
            hydrographs are going to be plots every spacing. The default is 5.
        fig_size: [tuple]
            default is (7, 5).
        x_labels: [bool, int]
            default is False.
        fmt: [string]
            format of the date. fmt="%Y-%m-%d %H:%M:%S".

        Returns
        -------
        None.
        """
        if start is None:
            start = self.first_day
        else:
            start = dt.datetime.strptime(start, fmt)

        if end is None:
            end = self.last_day
        else:
            end = dt.datetime.strptime(end, fmt)

        if from_xs is None:
            from_xs = self.first_xs

        if to_xs is None:
            to_xs = self.last_xs
            xss.append(to_xs)

        from_xs = self.xs_names.index(from_xs)
        to_xs = self.xs_names.index(to_xs)
        xs_list = self.xs_names[from_xs : to_xs + 1 : spacing]

        xs_list = xs_list + xss

        # to remove repeated XSs
        xs_list = list(set(xs_list))
        # extract the XS hydrographs
        for i in range(len(xs_list)):
            self.read_1d_results(xsid=xs_list[i])

        # xs_list = [self.first_xs] + xs_list + [self.last_xs]
        xs_list.sort()

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=fig_size)

        for i in range(len(xs_list)):
            ax.plot(
                self.xs_hydrograph.loc[start:end, xs_list[i]],
                label=f"XS-{xs_list[i]}",
                linewidth=line_width,
            ),  # color = xs_color,zorder=xs_order

        ax.legend(fontsize=10, loc="best")
        ax.set_xlabel("Time", fontsize=10)
        ax.set_ylabel("Discharge m3/s", fontsize=10)
        if not isinstance(x_labels, bool):
            start, end = ax.get_xlim()
            ax.xaxis.set_ticks(np.linspace(start, end, x_labels))

        plt.tight_layout()

        return fig, ax

    def read_us_hydrograph(
        self,
        from_day: int = None,
        to_day: int = None,
        path: str = None,
        date_format: str = "'%Y-%m-%d'",
    ):
        """readUSHydrograph.

            Read the hydrograph of the upstream reaches.

        Parameters
        ----------
        from_day: [int], optional
                the day you want to read the result from, the first day is 1 not zero. The default is ''.
        to_day: [int], optional
                the day you want to read the result to.
        path: [str], optional
            path to read the results from. if the path is not given, the customized_runs_path attribute for the river
            instance should be given. The default is ''.
        date_format: "TYPE, optional
            DESCRIPTION. The default is "'%Y-%m-%d'".

        Returns
        -------
        us_hydrographs: [dataframe attribute].
            dataframe contains the hydrograph of each of the upstream reachs with reach id as a column name, and a
            column 'total' contains the sum of all the hydrographs.
        """
        self.us_hydrographs = pd.DataFrame()

        if not path:
            path = self.customized_runs_path

        if len(self.upstream_node) > 1:
            # there is more than one upstream segment
            if isinstance(self.upstream_node, list):
                for i in range(len(self.upstream_node)):
                    node_id = self.upstream_node[i]
                    try:
                        self.us_hydrographs[node_id] = self._read_rrm_results(
                            self.version,
                            self.rrm_reference_index,
                            path,
                            node_id,
                            from_day,
                            to_day,
                            date_format,
                        )[node_id]
                        logger.info(f"the US hydrograph '{node_id}' has been read")
                    except FileNotFoundError:
                        msg = (
                            f" the Path - {path} does not contain the routed hydrographs for the the "
                            f"segment - {node_id}"
                        )
                        logger.info(msg)
                        return

            # there is one upstream segment
        elif self.upstream_node:
            node_id = self.upstream_node[0]
            try:
                self.us_hydrographs[node_id] = self._read_rrm_results(
                    self.version,
                    self.rrm_reference_index,
                    path,
                    node_id,
                    from_day,
                    to_day,
                    date_format,
                )[node_id]
                logger.info(f"the US hydrograph '{node_id}' has been read")
            except FileNotFoundError:
                logger.info(
                    f"The Path - {path} does not contain the routed hydrographs for the "
                    f"segment - {node_id}"
                )
                return
        else:
            logger.info(
                "the Segment Does not have any Upstream Segments, or you have "
                "not read the river network in the river instance"
            )
            return

        self.us_hydrographs["total"] = self.us_hydrographs.sum(axis=1)
        if not from_day:
            from_day = self.us_hydrographs.index[0]
        if not to_day:
            to_day = self.us_hydrographs.index[-1]

        start = self.reference_index.loc[from_day, "date"]
        end = self.reference_index.loc[to_day, "date"]

        self.us_hydrographs.index = pd.date_range(start, end, freq="D")

    def get_us_hydrograph(self, entire_river: River):
        """get_us_hydrograph.

        get_us_hydrograph method gets the sum of all the upstream hydrographs. whither it is routed inside the model
        or a boundary condition.

        Parameters
        ----------
        entire_river : [object]
            the object of the river.

        Returns
        -------
        us_hydrographs: [array].
            array of the hydrograph
        """
        self.us_hydrographs = np.zeros(shape=entire_river.no_time_steps)

        if len(self.upstream_node) > 1:
            # there is more than one upstream segment
            if isinstance(self.upstream_node, list):
                for i in range(len(self.upstream_node)):
                    node_id = self.upstream_node[i]
                    # get the order of the segment
                    entire_river.Segments.index(node_id)
                    self.us_hydrographs = (
                        self.us_hydrographs
                        + entire_river.routed_q[:, entire_river.Segments.index(node_id)]
                    )
            # there is one upstream segment
        elif self.upstream_node:
            node_id = self.upstream_node[0]
            entire_river.Segments.index(node_id)
            self.us_hydrographs = (
                self.us_hydrographs
                + entire_river.routed_q[:, entire_river.Segments.index(node_id)]
            )

        if not isinstance(self.BC, bool):
            self.us_hydrographs = self.us_hydrographs + self.BC.values.reshape(
                len(self.us_hydrographs)
            )

    def get_xs_geometry(self):
        """get_xs_geometry.

            get_xs_geometry calculates the area and perimeter for the cross-section highest and lowest point.

        Returns
        -------
        None.
        """
        area_per_low = np.zeros(shape=(self.xs_number, 2))
        area_per_high = np.zeros(shape=(self.xs_number, 2))
        for i in range(self.xs_number):
            geom = self.cross_sections.loc[i, :]
            h = min(geom["hl"], geom["hr"]) + geom["dbf"]
            coords = self.get_vortices(
                h,
                geom["hl"],
                geom["hr"],
                geom["bl"],
                geom["br"],
                geom["b"],
                geom["dbf"],
            )
            area_per_low[i, :] = self.polygon_geometry(coords)
            h = max(geom["hl"], geom["hr"]) + geom["dbf"]
            coords = self.get_vortices(
                h,
                geom["hl"],
                geom["hr"],
                geom["bl"],
                geom["br"],
                geom["b"],
                geom["dbf"],
            )
            area_per_high[i, :] = self.polygon_geometry(coords)
        self.AreaPerHigh = area_per_high[:, :]
        self.AreaPerLow = area_per_low[:, :]

    def get_flow(
        self,
        interface,
        from_day: Union[int, str] = "",
        to_day: Union[int, str] = "",
        date_format="%d_%m_%Y",
    ):
        """getFlow.

            Extract the lateral flow and boundary condition (if exist) time series of the segment from the whole
            river segment.

        Parameters
        ----------
        interface: [Interface object]
            You Have to create the interface object then read the laterals and the
            boundary conditions first.
        from_day: [string], optional
            the starting day. The default is ''.
        to_day: [string], optional
            the ending day. The default is ''.
        date_format: [string], optional
            the format of the given dates. The default is "%d_%m_%Y".

        Returns
        -------
        BC: [dataframe attribute].
            dataframe containing the boundary condition under a column by the name
            of the segment id.
        Laterals: [dataframe attribute].
            dataframe containing a column for each cross-section that has a lateral.
        """
        if hasattr(interface, "BC"):
            if not isinstance(interface.BC, DataFrame):
                raise ValueError(
                    "The boundary condition does not exist you have to read it first using the "
                    "'ReadBoundaryConditions' method in the interface model"
                )
        if hasattr(interface, "Laterals"):
            if not isinstance(interface.Laterals, DataFrame):
                raise ValueError(
                    "The Laterals do not exist you have to read it first using the 'ReadLaterals' method in the"
                    " interface model"
                )

        if from_day == "":
            from_day = interface.BC.index[0]
        else:
            from_day = dt.datetime.strptime(from_day, date_format)

        if to_day == "":
            to_day = interface.BC.index[-1]
        else:
            to_day = dt.datetime.strptime(to_day, date_format)

        # get the id of the boundary condition
        xs_as_set = set(self.xs_names)
        bc_list = [int(i) for i in interface.bc_table["xsid"].tolist()]
        bc_ids = list(xs_as_set.intersection(bc_list))

        if len(bc_ids) == 0:
            self.BC = False
        elif len(bc_ids) > 1:
            raise ValueError("There is more than one BC for this Reach-basin")
        else:
            self.BC = interface.BC.loc[from_day:to_day, bc_ids[0]].to_frame()

        if len(interface.laterals_table) > 0:
            self.laterals_table = [
                value
                for value in self.xs_names
                if value in interface.laterals_table["xsid"].tolist()
            ]
            self.Laterals = pd.DataFrame(
                index=pd.date_range(from_day, to_day, freq="D"),
                columns=self.laterals_table,
            )

            for i in self.laterals_table:
                self.Laterals.loc[:, i] = interface.Laterals.loc[from_day:to_day, i]

            self.Laterals["total"] = self.Laterals.sum(axis=1)
            # if the rrm hydrograph at the location of the hm or at the location of the rrm is read
            if isinstance(interface.routed_rrm, DataFrame):
                self.rrm_progression = pd.DataFrame(
                    index=pd.date_range(from_day, to_day, freq="D"),
                    columns=self.laterals_table,
                )
                for i in self.laterals_table:
                    self.rrm_progression.loc[:, i] = interface.routed_rrm.loc[
                        from_day:to_day, i
                    ]
        else:
            self.laterals_table = []
            self.Laterals = pd.DataFrame()

    def get_laterals(self, xsid: int):
        """get_laterals.

            get_laterals method gets the sum of the laterals of all the cross-sections in the reach upstream of a given
            xsid.

        Parameters
        ----------
        xsid : [integer]
            id of the cross-section.

        Returns
        -------
        dataframe:
            sum of the laterals of all the cross-sections in the reach
            upstream of a given xsid.
        """
        if not isinstance(self.laterals_table, list) and not isinstance(
            self.Laterals, DataFrame
        ):
            raise ValueError("please read the Laterals Table and the Laterals first")

        us_gauge = self.laterals_table[: bisect(self.laterals_table, xsid)]
        return self.Laterals[us_gauge].sum(axis=1).to_frame()

    def get_total_flow(self, gauge_xs: int):
        """getTotalFlow.

            getTotalFlow extracts all the laterals upstream of a certain xs and also extracts the Upstream/BC
            hydrograph.

        Parameters
        ----------
        gauge_xs : [integer]
            id of the cross-section.

        Returns
        -------
        total_flow: [dataframe attribute]
            dataframe containing the total upstream hydrograph for the location
            of the given xs, the column name is "total"
        """
        # Sum the laterals and the BC/US hydrograph
        if not isinstance(self.Laterals, DataFrame):
            raise ValueError("Please read the lateral flows first using the 'GetFlow'")

        if gauge_xs not in self.xs_names:
            raise ValueError(
                f"The given XS {gauge_xs} does not locate in the current river reach"
                f"First XS is {self.first_xs} and "
                f"Last XS is {self.last_xs}"
            )
        laterals = self.get_laterals(gauge_xs)
        try:
            s1 = laterals.index[0]
            e1 = laterals.index[-1]
        except IndexError:
            logger.info("there are no laterals for the given reach")
            return

        if isinstance(self.BC, DataFrame):
            s2 = self.BC.index[0]
            s = max(s1, s2)
            e2 = self.BC.index[-1]
            e = min(e1, e2)

            self.TotalFlow = pd.DataFrame(index=pd.date_range(s, e, freq="D"))
            self.TotalFlow.loc[s:e, "total"] = (
                laterals.loc[s:e, 0].values
                + self.BC.loc[s:e, self.BC.columns[0]].values
            )
            logger.info(f"Total flow for the XS-{gauge_xs} has been calculated")
        elif (
            isinstance(self.us_hydrographs, DataFrame) and len(self.us_hydrographs) > 0
        ):
            s2 = self.us_hydrographs.index[0]
            s = max(s1, s2)
            e2 = self.us_hydrographs.index[-1]
            e = min(e1, e2)

            self.TotalFlow = pd.DataFrame(index=pd.date_range(s, e, freq="D"))
            self.TotalFlow.loc[s:e, "total"] = (
                laterals.loc[s:e, 0].values
                + self.us_hydrographs.loc[s:e, "total"].values
            )
            logger.info(f"Total flow for the XS-{gauge_xs} has been calculated")
        else:
            logger.info(
                f"The US Hydrograph/BC of the given River reach-{self.id} is not read yet "
                "please use the 'ReadUSHydrograph' method to read it"
            )

    def h_to_q(self, q):
        """h_to_q.

            h_to_q method converts the discharge for a certain cross-section into water depth using the manning
            equation.

        Parameters
        ----------
        q : TYPE
            DESCRIPTION.

        Returns
        -------
        h : TYPE
            DESCRIPTION.
        """
        h = np.zeros(shape=(len(q)))

        for i in range(len(q)):
            # if Qbnd >  calculated Q for the highest depth in the table
            # highest depth = (dike height + 15 m)
            if q[i] > self.HQ[-1, 1]:
                # depth = highest depth
                h[i] = self.HQ[-1, 0]
                # if not calculate the Q for each discretized depth in the table
                # and take the one with the smallest difference from the q given
            else:
                qq = q[i]
                diff = abs(qq - self.HQ[:, 1])
                h[i] = self.HQ[np.argmin(diff), 0]

        return h

    plot_discharge_args = dict(
        Calib={"type": Any},
        gaugexs={"type": int},
        start={"type": str},
        end={"type": str},
        stationname={"type": int},
        gaugename={"type": [str, int]},
        segment_xs={"type": str},
        plotlaterals={"type": bool, "default": True},
        latcolor={"type": [str, tuple], "default": (0.3, 0, 0)},
        latorder={"type": int, "default": 4},
        latstyle={"type": int, "default": 9},
        plotus={"type": bool, "default": True},
        ushcolor={"type": [str, tuple], "default": "grey"},
        ushorder={"type": int, "default": 7},
        ushstyle={"type": int, "default": 7},
        plottotal={"type": bool, "default": True},
        totalcolor={"type": [str, tuple], "default": "k"},
        totalorder={"type": int, "default": 6},
        totalstyle={"type": int, "default": 4},
        specificxs={"type": [bool, int], "default": False},
        xscolor={"type": [str, tuple], "default": (164 / 255, 70 / 255, 159 / 255)},
        xsorder={"type": int, "default": 1},
        xslinestyle={"type": int, "default": 3},
        plotrrm={"type": bool, "default": True},
        rrmcolor={"type": [str, tuple], "default": "green"},
        rrmorder={"type": int, "default": 3},
        rrmlinestyle={"type": int, "default": 6},
        rrm2color={"type": [str, tuple], "default": (227 / 255, 99 / 255, 80 / 255)},
        rrm2linesytle={"type": int, "default": 8},
        plotgauge={"type": bool, "default": True},
        gaugecolor={"type": [str, tuple], "default": "#DC143C"},
        gaugeorder={"type": int, "default": 5},
        gaugestyle={"type": int, "default": 7},
        hmcolor={"type": [str, tuple], "default": "#004c99"},
        hmorder={"type": int, "default": 6},
        linewidth={"type": int, "default": 4},
        figsize={"type": tuple, "default": (6, 5)},
        fmt={"type": str, "default": "%Y-%m-%d"},
        xlabels={"type": [bool, int, list], "default": False},
        ylabels={"type": [bool, int, list], "default": False},
        # plotRRMProgression
        plothm={"type": bool, "default": True},
        rrmlinesytle={"type": int, "default": 8},
        # plotWL
        hmstyle={"type": int, "default": 6},
        legendsize={"type": Union[int, float], "default": 15},
        nxlabels={"type": int, "default": 4},
    )

    @class_method_parse(plot_discharge_args)
    def plot_q(
        self,
        calib,
        gauge_xs: int,
        start: str,
        end: str,
        station_name: int,
        gauge_name: Union[str, int],
        segment_xs: str,
        *args,
        **kwargs,
    ):
        """plot_q.

            plot the hydrograph at the gauge location for the hm, rrm (at two locations is availabe),
            sum of all laterals, upstream hydrograph, boundary condition hydrograph and the gauge time series.

        Parameters
        ----------
        calib: [Calibration object]
            DESCRIPTION.
        gauge_xs: integer
            the xsid of the gauge.
        start: [string]
            start date of the plot.
        end: [string]
            end date of the plot.
        station_name: [string]
            station name.
        gauge_name: TYPE
            DESCRIPTION.
        segment_xs: TYPE
            DESCRIPTION.
        kwargs:
            plotlaterals : TYPE, optional
                DESCRIPTION. The default is True.
            plotus : TYPE, optional
                DESCRIPTION. The default is True.
            specific_xs : TYPE, optional
                DESCRIPTION. The default is False.
            plotrrm : TYPE, optional
                DESCRIPTION. The default is True.
            plotgauge : TYPE, optional
                DESCRIPTION. The default is True.
            hmcolor : TYPE, optional
                DESCRIPTION. The default is "#004c99".
            gaugecolor : TYPE, optional
                DESCRIPTION. The default is "#DC143C".
            rrmcolor : TYPE, optional
                DESCRIPTION. The default is "green".
            latcolor : TYPE, optional
                DESCRIPTION. The default is (0.3,0,0).
            xscolor : TYPE, optional
                DESCRIPTION. The default is "grey".
            line_width : TYPE, optional
                DESCRIPTION. The default is 4.
            hmorder : TYPE, optional
                DESCRIPTION. The default is 6.
            gaugeorder : TYPE, optional
                DESCRIPTION. The default is 5.
            rrmorder : TYPE, optional
                DESCRIPTION. The default is 4.
            ushorder : TYPE, optional
                DESCRIPTION. The default is 2.
            xsorder : TYPE, optional
                DESCRIPTION. The default is 1.
            fmt: [string]
                format of the date. fmt="%Y-%m-%d %H:%M:%S"
            x_labels : [bool, int], optional
                DESCRIPTION. The default is False.
            ylabels : [bool, int], optional
                DESCRIPTION. The default is False.
            rrm2color: []
                Description
            gaugestyle: []
                Description
            rrm2linesytle: []
                Description
            ushstyle: []
                Description
            xslinestyle: []
                Description
            latorder: []
                Description
            ushcolor: []
                Description
            latstyle: []
                Description
            plottotal: [bool]
                default is True.
            totalcolor: [str, tuple]
                default is "k".
            totalorder: [int]
                default is 6.
            totalstyle: [int]
                default is 4.
            rrmlinestyle: [int]
                default is 6.
            fig_size: [tuple]
                default is (6, 5).

        Returns
        -------
        fig: TYPE
            DESCRIPTION.
        ax: TYPE
            DESCRIPTION.
        """
        start = dt.datetime.strptime(start, self.fmt)
        end = dt.datetime.strptime(end, self.fmt)

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=self.figsize)

        if self.xs_hydrograph is not None:
            # plot if you read the results using ther read_1d_results.
            try:
                ax.plot(
                    self.xs_hydrograph.loc[start:end, gauge_xs],
                    label="RIM",
                    zorder=self.hmorder,
                    linewidth=self.linewidth,
                    linestyle=Styles.get_line_style(6),
                    color=self.hmcolor,
                )
            except KeyError:
                logger.debug(
                    f"the xs given -{gauge_xs} - does not exist in the river reach"
                )

            # laterals
            if self.plotlaterals:
                try:
                    laterals = self.get_laterals(gauge_xs)
                except AssertionError:
                    logger.debug("please read the laterals first to be able to plot it")

                # BC

                if isinstance(self.BC, DataFrame):
                    ax.plot(
                        self.BC.loc[start:end, self.BC.columns[0]],
                        label="BC",
                        zorder=self.ushorder,
                        linewidth=self.linewidth,
                        linestyle=Styles.get_line_style(self.ushstyle),
                        color=self.ushcolor,
                    )
                # laterals
                if (
                    isinstance(self.laterals_table, list)
                    and len(self.laterals_table) > 0
                ):
                    ax.plot(
                        laterals.loc[start:end, 0],
                        label="laterals",
                        zorder=self.latorder,
                        linewidth=self.linewidth,
                        linestyle=Styles.get_line_style(self.latstyle),
                        color=self.latcolor,
                    )
                if self.plottotal:
                    # total flow
                    try:
                        ax.plot(
                            self.TotalFlow.loc[start:end, "total"],
                            label="US/BC + laterals",
                            zorder=self.totalorder,
                            linewidth=self.linewidth,
                            linestyle=Styles.get_line_style(self.totalstyle),
                            color=self.totalcolor,
                        )
                    except AttributeError:
                        logger.debug(
                            "there are no totalFlow for this reach please use the 'GetTotalFlow' method to create it"
                        )

            # US hydrograph
            if self.upstream_node != [] and self.plotus:
                try:
                    ax.plot(
                        self.us_hydrographs.loc[start:end, "total"],
                        label="US Hydrograph",
                        zorder=self.ushorder,
                        linewidth=self.linewidth,
                        linestyle=Styles.get_line_style(self.ushstyle),
                        color=self.ushcolor,
                    )
                except KeyError:
                    msg = (
                        "Please read the routed hydrograph of the upstream reaches using the "
                        "'read_us_hydrograph' method"
                    )

                    logger.debug(msg)

            # Gauge
            if self.plotgauge:
                # plot the gauge data
                ax.plot(
                    calib.q_gauges.loc[start:end, station_name],
                    label="Gauge",
                    linewidth=self.linewidth,
                    zorder=self.gaugeorder,
                    color=self.gaugecolor,
                    linestyle=Styles.get_line_style(self.gaugestyle),
                )

            # specific XS
            if not isinstance(self.specificxs, bool):
                # first extract the time series of the given xs
                self.read_1d_results(xsid=self.specificxs)
                # plot the xs
                ax.plot(
                    self.xs_hydrograph.loc[start:end, self.specificxs],
                    label="XS-" + str(self.specificxs),
                    zorder=self.xsorder,
                    linewidth=self.linewidth,
                    color=self.xscolor,
                    linestyle=Styles.get_line_style(self.xslinestyle),
                )
            # RRM
            if self.plotrrm:
                if isinstance(self.RRM, DataFrame):
                    try:
                        ax.plot(
                            self.RRM.loc[start:end, station_name],
                            label="mHM-RIM Loc",
                            zorder=self.rrmorder,
                            linewidth=self.linewidth,
                            linestyle=Styles.get_line_style(self.rrmlinestyle),
                            color=self.rrmcolor,
                        )
                    except KeyError:
                        logger.debug(
                            f" Station {gauge_name} does not have the first RRM discharge time series"
                        )

                if isinstance(self.RRM2, DataFrame):
                    try:
                        ax.plot(
                            self.RRM2.loc[start:end, station_name],
                            label="mHM-mHM Loc",
                            zorder=self.rrmorder,
                            linewidth=self.linewidth,
                            linestyle=Styles.get_line_style(self.rrm2linesytle),
                            color=self.rrm2color,
                        )
                    except KeyError:
                        logger.debug(
                            f" Station {gauge_name} does not have a second RRM discharge time series"
                        )

        elif isinstance(calib.calibration_q, DataFrame):
            # plot if you read the data using read_caliration_result.
            ax.plot(
                calib.calibration_q[segment_xs],
                label="RIM",
                zorder=3,
                linewidth=self.linewidth,
                linestyle=Styles.get_line_style(6),
                color=self.hmcolor,
            )
            # plot the gauge data
            ax.plot(
                calib.q_gauges.loc[
                    calib.calibration_q.index[0] : calib.calibration_q.index[-1],
                    station_name,
                ],
                label="Gauge-" + str(self.id),
                linewidth=self.linewidth,
                color=self.gaugecolor,
            )
            if self.plotrrm:
                ax.plot(
                    self.RRM.loc[
                        calib.calibration_q.index[0] : calib.calibration_q.index[-1],
                        station_name,
                    ],
                    label="RRM",
                )

        if not isinstance(self.xlabels, bool):
            start, end = ax.get_xlim()
            if isinstance(self.xlabels, int):
                ax.xaxis.set_ticks(np.linspace(start, end, self.xlabels))
            else:
                start = self.round(start, self.xlabels[0])
                end = self.round(end, self.xlabels[0])

                ax.yaxis.set_ticks(np.arange(start, end, self.xlabels[0]))

        if not isinstance(self.ylabels, bool):
            start, end = ax.get_ylim()
            if isinstance(self.ylabels, int):
                if start < 0:
                    start = 0
                ax.yaxis.set_ticks(np.linspace(start, end, self.ylabels))
            else:
                start = self.round(start, self.ylabels[0])
                end = self.round(end, self.ylabels[0])

                ax.yaxis.set_ticks(np.arange(start, end, self.ylabels[0]))

        ax.set_title("Discharge - " + gauge_name, fontsize=20)

        ax.legend(fontsize=12)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Discharge m3/s", fontsize=12)
        plt.tight_layout()

        return fig, ax

    @class_method_parse(plot_discharge_args)
    def plot_rrm_progression(self, specific_xs, start, end, *args, **kwargs):
        """plot_rrm_progression.

            plot the hydrograph at the gauge location for the hm, rrm (at two locations is available),
            sum of all laterals, upstream hydrograph, boundary condition hydrograph and the gauge time series.

        Parameters
        ----------
        specific_xs : integer
            the xsid of the gauge.
        start : [string]
            start date of the plot.
        end : [string]
            end date of the plot.
        kwargs:
            plotlaterals : TYPE, optional
                DESCRIPTION. The default is True.
            plotus : TYPE, optional
                DESCRIPTION. The default is True.
            specificxs : TYPE, optional
                DESCRIPTION. The default is False.
            hmcolor : TYPE, optional
                DESCRIPTION. The default is "#004c99".
            rrmcolor : TYPE, optional
                DESCRIPTION. The default is "green".
            latcolor : TYPE, optional
                DESCRIPTION. The default is (0.3,0,0).
            line_width : TYPE, optional
                DESCRIPTION. The default is 4.
            hmorder : TYPE, optional
                DESCRIPTION. The default is 6.
            rrmorder : TYPE, optional
                DESCRIPTION. The default is 4.
            ushorder : TYPE, optional
                DESCRIPTION. The default is 2.
            fmt: [string]
                format of the date. fmt="%Y-%m-%d %H:%M:%S"
            ushstyle: []
                Description
            latorder: []
                Description
            ushcolor: []
                Description
            latstyle: []
                Description
            x_labels: [int, bool]
                default is False.
            ylabels: [int, bool]
                default is False.
            fig_size: [tuple]
                default is (6, 5).
            rrmlinesytle: [int]
                default is 8
            plottotal: [bool]
                default is True.
            totalcolor: [str, tuple]
                default is "k".
            totalorder: [int]
                default is 6.
            totalstyle: [int]
                default is 11.
            plothm: [bool]
                default is True.

        Returns
        -------
        fig : TYPE
            DESCRIPTION.
        ax : TYPE
            DESCRIPTION.
        """
        start = dt.datetime.strptime(start, self.fmt)
        end = dt.datetime.strptime(end, self.fmt)

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=self.figsize)

        # laterals
        if self.plotlaterals:
            laterals = self.get_laterals(specific_xs)

            # BC
            if not isinstance(self.BC, bool):
                ax.plot(
                    self.BC.loc[start:end, self.BC.columns[0]],
                    label="BC",
                    zorder=self.ushorder,
                    linewidth=self.linewidth,
                    linestyle=Styles.get_line_style(self.ushstyle),
                    color=self.ushcolor,
                )
            # laterals
            if len(self.laterals_table) > 0:
                ax.plot(
                    laterals.loc[start:end, 0],
                    label="laterals Sum \n up to - XS-" + str(specific_xs),
                    zorder=self.latorder,
                    linewidth=self.linewidth,
                    linestyle=Styles.get_line_style(self.latstyle),
                    color=self.latcolor,
                )
            if self.plottotal:
                # total flow
                self.get_total_flow(specific_xs)
                ax.plot(
                    self.TotalFlow.loc[start:end, "total"],
                    label="US/BC \n+ laterals",
                    zorder=self.totalorder,
                    linewidth=self.linewidth,
                    linestyle=Styles.get_line_style(self.totalstyle),
                    color=self.totalcolor,
                )

        # US hydrograph
        if self.upstream_node != [] and self.plotus:
            ax.plot(
                self.us_hydrographs.loc[start:end, "total"],
                label="US Hydrograph",
                zorder=self.ushorder,
                linewidth=self.linewidth,
                linestyle=Styles.get_line_style(self.ushstyle),
                color=self.ushcolor,
            )

        # specific XS
        if self.plothm:
            # first extract the time series of the given xs
            self.read_1d_results(xsid=specific_xs)
            # plot the xs
            ax.plot(
                self.xs_hydrograph.loc[start:end, specific_xs],
                label="RIM",
                zorder=self.hmorder,
                linewidth=self.linewidth,
                linestyle=Styles.get_line_style(6),
                color=self.hmcolor,
            )
        # RRM
        # if plotrrm:
        if hasattr(self, "routed_rrm"):
            try:
                ax.plot(
                    self.rrm_progression.loc[start:end, specific_xs],
                    label="mHM",
                    zorder=self.rrmorder,
                    linewidth=self.linewidth,
                    linestyle=Styles.get_line_style(self.rrmlinesytle),
                    color=self.rrmcolor,
                )
            except KeyError:
                logger.debug(
                    " XS " + str(specific_xs) + "does not exist in the  'routed_rrm'"
                )
        else:
            msg = (
                "please read the RRM hydrographs using the 'ReadRRMProgression'"
                "in the interface module"
            )
            logger.debug(msg)

        if not isinstance(self.xlabels, bool):
            start, end = ax.get_xlim()
            if isinstance(self.xlabels, int):
                ax.xaxis.set_ticks(np.linspace(start, end, self.xlabels))

        if not isinstance(self.ylabels, bool):
            start, end = ax.get_ylim()
            if isinstance(self.ylabels, int):
                if start < 0:
                    start = 0
                ax.yaxis.set_ticks(np.linspace(start, end, self.ylabels))
            else:
                start = self.round(start, self.ylabels[0])
                end = self.round(end, self.ylabels[0])

                ax.yaxis.set_ticks(np.arange(start, end, self.ylabels[0]))

        ax.set_title("XS - " + str(specific_xs), fontsize=20)

        ax.legend(fontsize=12)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Discharge m3/s", fontsize=12)
        plt.tight_layout()

        return fig, ax

    def calculate_q_metrics(
        self,
        calib,
        station_name: int,
        gauge_xs: int,
        filter: bool = False,
        start: str = "",
        end: str = "",
        fmt: str = "%Y-%m-%d",
    ):
        """calculate_q_metrics.

            calculate_q_metrics calculates the performance metrics for the discharge time series.

        Parameters
        ----------
        calib : TYPE
            DESCRIPTION.
        station_name : TYPE
            DESCRIPTION.
        gauge_xs : [int]
            DESCRIPTION.
        start : [str]
            DESCRIPTION.
        end : [str]
            DESCRIPTION.
        filter: TYPE, optional
            DESCRIPTION. The default is False.
        fmt: [string]
            format of the date. fmt="%Y-%m-%d %H:%M:%S"

        Returns
        -------
        rmse: [float]
            root-mean-square error.
        kge: [float]
            Kling-gupta metric.
        wb: [float]
            water balance metrix.
        nsehf: [float]
            Nash-sutcliffe for high values metric.
        nse: [float]
            Nash-sutcliffe metric.
        """
        q_hm = pd.DataFrame()

        try:
            gauge_start = calib.hm_gauges.loc[
                calib.hm_gauges["xsid"] == gauge_xs, "Qstart"
            ].values[0]
            gauge_end = calib.hm_gauges.loc[
                calib.hm_gauges["xsid"] == gauge_xs, "Qend"
            ].values[0]
        except IndexError:
            logger.debug("The XS you provided does not exist in the hm_gauges")
            return

        if filter:
            start = dt.datetime.strptime(start, fmt)
            end = dt.datetime.strptime(end, fmt)
            # get the latest date of the filter date and the first date in the result
            # get the earliest date of the end and the last date in the result
            st2 = max(gauge_start, start, self.first_day_results)
            end2 = min(gauge_end, end, self.last_day)

            # get the observed discharge
            q_obs = calib.q_gauges.loc[st2:end2, station_name]

            # resample the time series to average daily
            ind = pd.date_range(
                self.first_day_results, self.last_day + dt.timedelta(days=1), freq="h"
            )[:-1]

            q = self.results_1d[self.results_1d["xs"] == self.last_xs]
            q.index = ind
            q_hm["q"] = q["q"].resample("D").mean()
            q_hm["q"] = q_hm.loc[st2:end2, "q"]

            # try:
            #     sub.Resample(gaugexs, 'q', starti, endi, delete=True)
            # except:
            #     sub.Resample(gaugexs, 'q', starti, endi, delete=False)
            # q_hm['q']  = sub.resampled_q[gaugexs][:]
            # q_hm.index = pd.date_range(st2, end2)

        else:
            st2 = max(gauge_start, self.first_day_results)
            end2 = min(gauge_end, self.last_day)
            # get the observed discharge
            q_obs = calib.q_gauges.loc[st2:end2, station_name]

            # resample the time series to average daily
            ind = pd.date_range(
                self.first_day_results, self.last_day + dt.timedelta(days=1), freq="h"
            )[:-1]
            q = self.results_1d[self.results_1d["xs"] == self.last_xs]
            q.index = ind
            q_hm["q"] = q["q"].resample("D").mean()
            q_hm["q"] = q_hm.loc[st2:end2, "q"]

            # old
            # q_hm['q'] = sub.results_1d['q'][sub.results_1d['xs'] == gaugexs][sub.results_1d['hour'] == 24][:]
            # q_hm.index = pd.date_range(st2, end2)
        qsim = q_hm.loc[st2:end2, "q"].tolist()
        rmse = round(metrics.rmse(q_obs, qsim), 0)
        kge = round(metrics.kge(q_obs, qsim), 2)
        wb = round(metrics.wb(q_obs, qsim), 0)
        nsehf = round(metrics.nse_hf(q_obs, qsim), 2)
        nse = round(metrics.nse(q_obs, qsim), 2)
        logger.debug("--------------------")
        logger.debug("RMSE = " + str(rmse))
        logger.debug("KGE = " + str(kge))
        logger.debug("WB = " + str(wb))
        logger.debug("NSEHF = " + str(nsehf))
        logger.debug("NSE = " + str(nse))

        return rmse, kge, wb, nsehf, nse

    @class_method_parse(plot_discharge_args)
    def plot_wl(
        self,
        calib,
        start: str,
        end: str,
        gauge_xs: int,
        station_name: str,
        gauge_name: str,
        *args,
        **kwargs,
        # gaugecolor: Union[tuple, str] = "#DC143C",
        # hmcolor: Union[tuple, str] = "#004c99",
        # line_width: Union[int, float] = 2,
        # hmorder: int = 1,
        # gaugeorder: int = 0,
        # hmstyle: int = 6,
        # gaugestyle: int = 0,
        # plotgauge=True,
        # fmt: str = "%Y-%m-%d",
        # legendsize: Union[int, float] = 15,
        # fig_size: tuple = (6, 5),
        # xlabels_number: int = 4,
    ):
        """Plot water level surface.

        plot water level hydrograph

        Parameters
        ----------
        calib : TYPE
            DESCRIPTION.
        start : TYPE
            DESCRIPTION.
        end : TYPE
            DESCRIPTION.
        gauge_xs : TYPE
            DESCRIPTION.
        station_name : TYPE
            DESCRIPTION.
        gauge_name : TYPE
            DESCRIPTION.
        kwargs:
            gaugecolor : TYPE, optional
                DESCRIPTION. The default is "#DC143C".
            hmcolor : TYPE, optional
                DESCRIPTION. The default is "#004c99".
            line_width : TYPE, optional
                DESCRIPTION. The default is 2.
            hmorder : TYPE, optional
                DESCRIPTION. The default is 1.
            gaugeorder : TYPE, optional
                DESCRIPTION. The default is 0.
            plotgauge : TYPE, optional
                DESCRIPTION. The default is True.
            fmt: [string]
                format of the date. fmt="%Y-%m-%d %H:%M:%S"
            hmstyle: [int]
                default is 6
            gaugestyle: [int]
                default is 0.
            legendsize: [int, float]
                default is 15.
            fig_size: tuple=(6, 5),
            xlabels_number: [int]
                default is 4.

        Returns
        -------
        ax : TYPE
            DESCRIPTION.
        """
        start = dt.datetime.strptime(start, self.fmt)
        end = dt.datetime.strptime(end, self.fmt)

        if self.plotgauge:
            gauge_start = calib.hm_gauges[calib.hm_gauges["xsid"] == gauge_xs][
                "WLstart"
            ]
            gauge_end = calib.hm_gauges[calib.hm_gauges["xsid"] == gauge_xs]["WLend"]

            try:
                gauge_start = gauge_start.values[0]
                gauge_end = gauge_end.values[0]

                if gauge_start > start and gauge_start > end:
                    logger.debug(
                        f"Available data for the gauge starts from {gauge_start}"
                    )
                    logger.debug(
                        f"The period you provided is between {start} and {end}"
                    )
                    plot_gauge = False
                elif gauge_end < start and gauge_start < end:
                    logger.debug(
                        f"Available data for the gauge starts from {gauge_end}"
                    )
                    logger.debug("Out of Gauge dates")
                    plot_gauge = False
            except (IndexError, TypeError):
                logger.debug("The XS you provided does not exist in the hm_gauges")
                plot_gauge = False

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=self.figsize)

        # extract the water levels at the gauge cross-section
        self.extract_xs(gauge_xs)

        ax.plot(
            self.xs_water_level.loc[start:end, gauge_xs],
            label="Hydraulic model",
            zorder=self.hmorder,
            linewidth=self.linewidth,
            color=self.hmcolor,
            linestyle=Styles.get_line_style(self.hmstyle),
        )

        if self.plotgauge:
            ax.plot(
                calib.wl_gauges.loc[start:end, station_name],
                label="Gauge",
                zorder=self.gaugeorder,
                linewidth=self.linewidth,
                color=self.gaugecolor,
                linestyle=Styles.get_line_style(self.gaugestyle),
            )

        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.linspace(start, end, self.nxlabels))

        ax.set_title(f"Water Level - {gauge_name}", fontsize=20)
        plt.legend(fontsize=self.legendsize)
        ax.set_xlabel("Time", fontsize=15)
        ax.set_ylabel("Water Level m", fontsize=15)
        plt.tight_layout()

        return fig, ax

    def calculate_wl_metrics(
        self,
        calib,
        station_name: int,
        gauge_xs: int,
        filter: bool = False,
        start: Union[dt.datetime, str] = "",
        end: Union[dt.datetime, str] = "",
        fmt: str = "%Y-%m-%d",
    ):
        """CalculateWLMetrics. calculate the performance metrics for the water level time series.

        fmt: [string]
            format of the date. fmt="%Y-%m-%d %H:%M:%S"
        """
        try:
            gauge_start = calib.hm_gauges[calib.hm_gauges["xsid"] == gauge_xs][
                "WLstart"
            ].values[0]
            gauge_end = calib.hm_gauges[calib.hm_gauges["xsid"] == gauge_xs][
                "WLend"
            ].values[0]
        except IndexError:
            logger.debug("The XS you provided does not exist in the hm_gauges")
            return

        if isinstance(gauge_start, int):
            logger.debug("No water level data for this river reach")
            return

        if filter:
            if isinstance(start, str):
                start = dt.datetime.strptime(start, fmt)
            if isinstance(end, str):
                end = dt.datetime.strptime(end, fmt)

            st2 = max(gauge_start, start, self.first_day_results)
            end2 = min(gauge_end, end, self.last_day)
            # observed
            obs = np.array(calib.wl_gauges.loc[st2:end2, station_name])

            # RIM
            ind = pd.date_range(
                self.first_day_results, self.last_day + dt.timedelta(days=1), freq="h"
            )[:-1]
            mod = self.results_1d[self.results_1d["xs"] == self.last_xs]
            mod.index = ind
            mod = mod["wl"].resample("D").mean()
            mod = mod.loc[st2:end2]

            # RIM
            # try:
            #     sub.Resample(gaugexs, 'wl', River.DateToIndex(st2),
            #                   River.DateToIndex(end2), delete = True)
            # except:
            #     sub.Resample(gaugexs, 'wl', River.DateToIndex(st2),
            #                   River.DateToIndex(end2), delete = False)
            # series1 = np.array(sub.resampled_wl[gaugexs])
        else:
            st2 = max(gauge_start, self.first_day_results)
            end2 = min(gauge_end, self.last_day)
            # Observed
            obs = np.array(calib.wl_gauges.loc[st2:end2, station_name])

            # RIM
            ind = pd.date_range(
                self.first_day_results, self.last_day + dt.timedelta(days=1), freq="h"
            )[:-1]
            mod = self.results_1d[self.results_1d["xs"] == gauge_xs]
            mod.index = ind
            mod = mod["wl"].resample("D").mean()
            mod = mod.loc[st2:end2]

            # RIM
            # sub.Resample(gaugexs, 'wl', River.DateToIndex(st2),
            #               River.DateToIndex(end2), delete = False)
            # series1 = np.array(sub.resampled_wl[gaugexs])

        if len(obs) != len(mod) or len(mod) == 0:
            logger.debug(
                "Available data for the gauge starts from "
                + str(gauge_start)
                + " To "
                + str(gauge_end)
            )
            return

        mbe = round(metrics.mbe(obs, mod), 2)
        mae = round(metrics.mae(obs, mod), 2)
        rmse = round(metrics.rmse(obs, mod), 2)
        kge = round(metrics.kge(obs, mod), 2)
        nsehf = round(metrics.nse_hf(obs, mod), 2)
        nse = round(metrics.nse(obs, mod), 2)

        logger.debug("rmse= " + str(rmse))
        logger.debug("kge= " + str(kge))
        logger.debug("nsehf= " + str(nsehf))
        logger.debug("nse= " + str(nse))
        logger.debug("mbe= " + str(mbe))
        logger.debug("mae= " + str(mae))

        return mbe, mae, rmse, kge, nsehf, nse

    def plot_bc(self, date: str, fmt: str = "%Y-%m-%d"):
        """PlotBC. plot the boundary condition discharge and water depth.

        Parameters
        ----------
        date : TYPE
            DESCRIPTION.
        fmt: [string] optional
            format of the date. fmt="%Y-%m-%d %H:%M:%S", default is fmt="%Y-%m-%d"

        Returns
        -------
        None.
        """
        date = dt.datetime.strptime(date, fmt)

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.h_bc_1min.loc[date])
        ax1.set_xlabel("Date", fontsize=15)
        ax1.set_ylabel("H", fontsize=15)
        ax1.set_xlim(0, 1440)
        ax2.plot(self.q_bc_1min.loc[date])
        ax2.set_ylabel("Q", fontsize=15)
