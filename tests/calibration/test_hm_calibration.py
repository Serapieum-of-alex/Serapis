from typing import List

from geopandas import GeoDataFrame
from matplotlib.figure import Figure
from pandas.core.frame import DataFrame

import serapis.calibration as RC


def test_create_calibration_instance(
    version: int,
):
    RC.Calibration("HM", version=version)


def test_ReadGaugesTable(gauges_table_path: str):
    Calib = RC.Calibration("HM", version=3)
    Calib.read_gauges_table(gauges_table_path)
    assert len(Calib.hm_gauges) == 3 and len(Calib.hm_gauges.columns) == 10


def test_ReadObservedQ(
    gauges_table_path: str,
    ReadObservedQ_Path: str,
    dates: list,
    nodatavalu: int,
    gauge_date_format: str,
    version: int,
    test_time_series_length: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.read_gauges_table(gauges_table_path)
    Calib.read_observed_q(
        ReadObservedQ_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )

    assert (
        len(Calib.q_gauges) == test_time_series_length
        and len(Calib.q_gauges.columns) == 3
        and len(Calib.hm_gauges.columns) == 12
    )


def test_ReadObservedWL(
    gauges_table_path: str,
    ReadObservedWL_Path: str,
    dates: list,
    nodatavalu: int,
    gauge_date_format: str,
    version: int,
    test_time_series_length: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.read_gauges_table(gauges_table_path)
    Calib.read_observed_wl(
        ReadObservedWL_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )
    assert (
        len(Calib.wl_gauges) == test_time_series_length
        and len(Calib.wl_gauges.columns) == 3
        and len(Calib.hm_gauges.columns) == 12
    )


def test_CalculateProfile(
    version: int,
    segment3: int,
    river_cross_section_path: str,
    river_network_path: str,
    calibrateProfile_DS_bedlevel: float,
    calibrateProfile_mn: float,
    calibrateProfile_slope: float,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.read_xs(river_cross_section_path)
    Calib.calculate_profile(
        segment3,
        calibrateProfile_DS_bedlevel,
        calibrateProfile_mn,
        calibrateProfile_slope,
    )

    assert (
        Calib.cross_sections.loc[Calib.cross_sections["id"] == 3, "gl"].tolist()[-1]
        == calibrateProfile_DS_bedlevel
    )


def test_SmoothMaxSlope(
    version: int,
    river_cross_section_path: str,
    segment3: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.read_xs(river_cross_section_path)
    Calib.smooth_max_slope(segment3)


def test_SmoothBedLevel(
    version: int,
    river_cross_section_path: str,
    segment3: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.read_xs(river_cross_section_path)
    Calib.smooth_bed_level(segment3)


def test_SmoothDikeLevel(
    version: int,
    river_cross_section_path: str,
    segment3: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.read_xs(river_cross_section_path)
    Calib.smooth_dike_level(segment3)


def test_DownWardBedLevel(
    version: int,
    river_cross_section_path: str,
    segment3: int,
    DownWardBedLevel_height: float,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.read_xs(river_cross_section_path)
    Calib.downward_bed_level(segment3, DownWardBedLevel_height)


def test_SmoothBankLevel(
    version: int,
    river_cross_section_path: str,
    segment3: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.read_xs(river_cross_section_path)
    Calib.smooth_bank_level(segment3)


def test_SmoothFloodplainHeight(
    version: int,
    river_cross_section_path: str,
    segment3: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.read_xs(river_cross_section_path)
    Calib.smooth_floodplain_height(segment3)


def test_SmoothBedWidth(
    version: int,
    river_cross_section_path: str,
    segment3: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.read_xs(river_cross_section_path)
    Calib.smooth_bed_width(segment3)


def test_CheckFloodplain(
    version: int,
    river_cross_section_path: str,
    segment3: int,
):
    Calib = RC.Calibration("HM", version=version)
    Calib.read_xs(river_cross_section_path)
    Calib.check_floodplain()


def test_ReadRRM(
    gauges_table_path: str,
    rrmpath: str,
    test_time_series_length: int,
    rrmgauges: List[int],
):
    Calib = RC.Calibration("HM", version=3)
    Calib.read_gauges_table(gauges_table_path)
    Calib.read_rrm(rrmpath, fmt="'%Y-%m-%d'")
    assert len(Calib.q_rrm) == test_time_series_length and len(
        Calib.q_rrm.columns
    ) == len(rrmgauges)
    assert all(elem in Calib.q_rrm.columns.to_list() for elem in rrmgauges)


def test_ReadHMQ(
    gauges_table_path: str,
    hm_separated_q_results_path: str,
    test_time_series_length: int,
    rrmgauges: List[int],
):
    Calib = RC.Calibration("HM", version=3)
    Calib.read_gauges_table(gauges_table_path)
    Calib.read_hm_discharge(hm_separated_q_results_path, fmt="'%Y-%m-%d'")
    assert len(Calib.q_hm) == test_time_series_length and len(
        Calib.q_hm.columns
    ) == len(rrmgauges)
    assert all(elem in Calib.q_hm.columns.to_list() for elem in rrmgauges)


def test_ReadHMWL(
    gauges_table_path: str,
    hm_separated_wl_results_path: str,
    test_time_series_length: int,
    rrmgauges: List[int],
):
    Calib = RC.Calibration("HM", version=3)
    Calib.read_gauges_table(gauges_table_path)
    Calib.read_hm_water_level(hm_separated_wl_results_path, fmt="'%Y-%m-%d'")
    assert len(Calib.wl_hm) == test_time_series_length and len(
        Calib.wl_hm.columns
    ) == len(rrmgauges)
    assert all(elem in Calib.wl_hm.columns.to_list() for elem in rrmgauges)


class Test_GetAnnualMax:
    def test_option_1(
        self,
        gauges_table_path: str,
        ObservedQ_long_ts_Path: str,
        ObservedQ_long_ts_dates: list,
        nodatavalu: int,
        gauge_long_ts_date_format: str,
        ObservedQ_long_ts_len: int,
        gauges_numbers: int,
    ):
        """test extracting max annual observed discharge."""
        Calib = RC.Calibration("HM", version=3)
        Calib.read_gauges_table(gauges_table_path)
        Calib.read_observed_q(
            ObservedQ_long_ts_Path,
            ObservedQ_long_ts_dates[0],
            ObservedQ_long_ts_dates[1],
            nodatavalu,
            gauge_date_format=gauge_long_ts_date_format,
        )
        # RIM.ReadObservedQ(ObservedPath, GRDCStartDate, GRDCEndDate, NoValue)

        Calib.get_annual_max(option=1)
        assert len(Calib.annual_max_obs_q) == ObservedQ_long_ts_len
        assert len(Calib.annual_max_obs_q.columns) == gauges_numbers

    def test_option_3(
        self,
        gauges_table_path: str,
        rrm_long_ts_number: int,
        gauges_numbers: int,
        rrmpath_long_ts: str,
    ):
        """test extracting max annual hydrologic model simulated discharge."""
        Calib = RC.Calibration("HM", version=3)
        Calib.read_gauges_table(gauges_table_path)
        Calib.read_rrm(rrmpath_long_ts, fmt="'%Y-%m-%d'")

        Calib.get_annual_max(option=3)
        assert len(Calib.annual_max_rrm) == rrm_long_ts_number
        assert len(Calib.annual_max_rrm.columns) == gauges_numbers

    def test_option_4(
        self,
        gauges_table_path: str,
        hm_long_ts_number: int,
        gauges_numbers: int,
        hm_separated_results_q_long_ts_path,
    ):
        """test extracting max annual hydraulic model simulated discharge."""
        Calib = RC.Calibration("HM", version=3)
        Calib.read_gauges_table(gauges_table_path)
        Calib.read_hm_discharge(hm_separated_results_q_long_ts_path, fmt="'%Y-%m-%d'")

        Calib.get_annual_max(option=4)
        assert len(Calib.annual_max_hm_q) == hm_long_ts_number
        assert len(Calib.annual_max_hm_q.columns) == gauges_numbers


def test_HMvsRRM(
    gauges_table_path: str,
    rrmpath: str,
    hm_separated_q_results_path: str,
    test_time_series_length: int,
    rrmgauges: List[int],
    Metrics_table_columns: List[str],
):
    Calib = RC.Calibration("HM", version=3)
    Calib.read_gauges_table(gauges_table_path)
    Calib.read_hm_discharge(hm_separated_q_results_path, fmt="'%Y-%m-%d'")
    Calib.read_rrm(rrmpath, fmt="'%Y-%m-%d'")
    Calib.hm_vs_rrm()
    assert isinstance(Calib.metrics_hm_vs_rrm, DataFrame) and isinstance(
        Calib.metrics_hm_vs_rrm, GeoDataFrame
    )
    assert len(Calib.metrics_hm_vs_rrm) == 3
    assert all(
        Calib.metrics_hm_vs_rrm.index
        == Calib.hm_gauges.loc[:, Calib.gauge_id_col].to_list()
    )
    assert all(Calib.metrics_hm_vs_rrm.columns == Metrics_table_columns)


def test_RRMvsObserved(
    gauges_table_path: str,
    ReadObservedQ_Path: str,
    dates: list,
    nodatavalu: int,
    gauge_date_format: str,
    rrmpath: str,
    Metrics_table_columns: List[str],
):
    Calib = RC.Calibration("HM", version=3, start=dates[0])
    Calib.read_gauges_table(gauges_table_path)
    Calib.read_observed_q(
        ReadObservedQ_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )
    Calib.read_rrm(rrmpath, fmt="'%Y-%m-%d'")
    Calib.rrm_vs_observed()
    assert isinstance(Calib.metrics_rrm_vs_obs, DataFrame) and isinstance(
        Calib.metrics_rrm_vs_obs, GeoDataFrame
    )
    assert len(Calib.metrics_rrm_vs_obs) == 3
    assert all(
        Calib.metrics_rrm_vs_obs.index
        == Calib.hm_gauges.loc[:, Calib.gauge_id_col].to_list()
    )
    assert all(
        elem in Calib.metrics_rrm_vs_obs.columns for elem in Metrics_table_columns
    )


def test_HMQvsObserved(
    gauges_table_path: str,
    ReadObservedQ_Path: str,
    dates: list,
    nodatavalu: int,
    gauge_date_format: str,
    hm_separated_q_results_path: str,
    Metrics_table_columns: List[str],
):
    Calib = RC.Calibration("HM", version=3, start=dates[0])
    Calib.read_gauges_table(gauges_table_path)
    Calib.read_observed_q(
        ReadObservedQ_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )
    Calib.read_hm_discharge(hm_separated_q_results_path, fmt="'%Y-%m-%d'")
    Calib.hm_vs_observed_discharge()
    assert isinstance(Calib.metrics_hm_q_vs_obs, DataFrame) and isinstance(
        Calib.metrics_hm_q_vs_obs, GeoDataFrame
    )
    assert len(Calib.metrics_hm_q_vs_obs) == 3
    assert all(
        Calib.metrics_hm_q_vs_obs.index
        == Calib.hm_gauges.loc[:, Calib.gauge_id_col].to_list()
    )
    assert all(
        elem in Calib.metrics_hm_q_vs_obs.columns for elem in Metrics_table_columns
    )


def test_HMWLvsObserved(
    gauges_table_path: str,
    ReadObservedWL_Path: str,
    dates: list,
    nodatavalu: int,
    gauge_date_format: str,
    hm_separated_wl_results_path: str,
    Metrics_table_columns: List[str],
):
    Calib = RC.Calibration("HM", version=3, start=dates[0])
    Calib.read_gauges_table(gauges_table_path)
    Calib.read_observed_wl(
        ReadObservedWL_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )
    Calib.read_hm_water_level(hm_separated_wl_results_path, fmt="'%Y-%m-%d'")
    Calib.hm_vs_observed_water_level()
    assert isinstance(Calib.metrics_hm_wl_vs_obs, DataFrame) and isinstance(
        Calib.metrics_hm_wl_vs_obs, GeoDataFrame
    )
    assert len(Calib.metrics_hm_wl_vs_obs) == 3
    assert all(
        Calib.metrics_hm_wl_vs_obs.index
        == Calib.hm_gauges.loc[:, Calib.gauge_id_col].to_list()
    )
    assert all(
        elem in Calib.metrics_hm_wl_vs_obs.columns for elem in Metrics_table_columns
    )


def test_InspectGauge(
    gauges_table_path: str,
    ReadObservedWL_Path: str,
    ReadObservedQ_Path: str,
    rrmpath: str,
    dates: list,
    nodatavalu: int,
    gauge_date_format: str,
    hm_separated_q_results_path: str,
    hm_separated_wl_results_path: str,
    Metrics_table_columns: List[str],
    InspectGauge_sub_id: int,
):
    Calib = RC.Calibration("HM", version=3, start=dates[0])
    Calib.read_gauges_table(gauges_table_path)
    Calib.read_observed_wl(
        ReadObservedWL_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )
    Calib.read_hm_water_level(hm_separated_wl_results_path, fmt="'%Y-%m-%d'")
    Calib.hm_vs_observed_water_level()
    Calib.read_observed_q(
        ReadObservedQ_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )
    Calib.read_hm_discharge(hm_separated_q_results_path, fmt="'%Y-%m-%d'")
    Calib.hm_vs_observed_discharge()

    Calib.read_rrm(rrmpath, fmt="'%Y-%m-%d'")
    Calib.hm_vs_rrm()
    Calib.rrm_vs_observed()
    gaugei = 0
    summary, fig, ax = Calib.Inspect_gauge(InspectGauge_sub_id, gaugei=gaugei)
    assert isinstance(fig, Figure)
    assert isinstance(summary, DataFrame)
    assert all(elem in summary.columns.to_list() for elem in Metrics_table_columns)


def test_SaveMetices(
    gauges_table_path: str,
    ReadObservedWL_Path: str,
    ReadObservedQ_Path: str,
    rrmpath: str,
    dates: list,
    nodatavalu: int,
    gauge_date_format: str,
    hm_separated_q_results_path: str,
    hm_separated_wl_results_path: str,
    Metrics_table_columns: List[str],
    hm_saveto: str,
):
    Calib = RC.Calibration("HM", version=3, start=dates[0])
    Calib.read_gauges_table(gauges_table_path)
    Calib.read_observed_wl(
        ReadObservedWL_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )
    Calib.read_hm_water_level(hm_separated_wl_results_path, fmt="'%Y-%m-%d'")
    Calib.hm_vs_observed_water_level()
    Calib.read_observed_q(
        ReadObservedQ_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )
    Calib.read_hm_discharge(hm_separated_q_results_path, fmt="'%Y-%m-%d'")
    Calib.hm_vs_observed_discharge()

    Calib.read_rrm(rrmpath, fmt="'%Y-%m-%d'")
    Calib.hm_vs_rrm()
    Calib.rrm_vs_observed()
    Calib.save_metices(hm_saveto)
