"""test hm_river.py module."""
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.frame import DataFrame
from matplotlib.figure import Figure

import serapis.calibration as rc
from serapis.river import River, Reach
from serapis.event import Event
from serapis.interface import Interface


def test_create_river_instance(dates: list, rrm_start: str, version: int):
    assert River("HM", version=version, start=dates[0], rrm_start=rrm_start)


def test_read_slope_method(version: int, slope_path: str):
    rivers = River("HM", version=version)
    rivers.read_slope(slope_path)
    assert len(rivers.slope) == 2 and len(rivers.slope.columns) == 2


def test_read_cross_sections_method(
    version: int, river_cross_section_path: str, xs_total_no: int, xs_col_no: int
):
    rivers = River("HM", version=version)
    rivers.read_xs(river_cross_section_path)
    assert (
        len(rivers.cross_sections) == xs_total_no
        and len(rivers.cross_sections.columns) == xs_col_no
    )


def test_read_river_network_method(version: int, river_network_path: str):
    rivers = River("HM", version=version)
    rivers.read_river_network(river_network_path)
    assert len(rivers.river_network) == 3 and len(rivers.river_network.columns) == 3


def test_create_sub_instance(
    segment1: int,
    version: int,
    river_cross_section_path: str,
    river_network_path: str,
    slope_path: str,
    create_sub_instance_first_xs: int,
    create_sub_instance_last_xs: int,
):
    rivers = River("HM", version=version)
    rivers.read_xs(river_cross_section_path)
    rivers.read_river_network(river_network_path)
    rivers.read_slope(slope_path)
    sub = Reach(segment1, rivers)
    assert (
        sub.first_xs == create_sub_instance_first_xs
        and sub.last_xs == create_sub_instance_last_xs
    )
    assert sub.slope


def test_sub_get_flow(
    segment1: int,
    version: int,
    river_cross_section_path: str,
    river_network_path: str,
    slope_path: str,
    dates: list,
    interface_bc_path: str,
    interface_bc_folder: str,
    interface_bc_date_format: str,
    interface_laterals_table_path: str,
    interface_laterals_folder: str,
    interface_laterals_date_format: str,
    sub_get_flow_lateral_table,
    test_time_series_length: int,
):
    rivers = River("HM", version=version)
    rivers.read_xs(river_cross_section_path)
    rivers.read_river_network(river_network_path)
    rivers.read_slope(slope_path)
    sub = Reach(segment1, rivers)

    interface = Interface("Rhine", start=dates[0])
    interface.readBoundaryConditionsTable(interface_bc_path)
    interface.read_boundary_conditions(
        path=interface_bc_folder, date_format=interface_bc_date_format
    )

    interface.read_xs(river_cross_section_path)
    interface.readLateralsTable(interface_laterals_table_path)
    interface.readLaterals(
        path=interface_laterals_folder, date_format=interface_laterals_date_format
    )

    sub.get_flow(interface)

    assert (
        len(sub.BC) == len(sub.Laterals) == test_time_series_length
        and len(sub.BC.columns) == 1
        and len(sub.Laterals.columns) == 4
    )
    assert all(elem in sub.laterals_table for elem in sub_get_flow_lateral_table)


class TestRead1DResult:
    def test_read_complete_file(
        self,
        version: int,
        river_cross_section_path: str,
        segment1: int,
        read_1d_result_path: str,
        read_1d_result_xs_id: int,
        test_time_series_length: int,
    ):
        rivers = River("HM", version=version)
        rivers.one_d_result_path = read_1d_result_path
        # rivers.results_paths = {"one_d_result_path": Read1DResult_path}
        rivers.read_xs(river_cross_section_path)
        sub = Reach(segment1, rivers)
        sub.read_1d_results(extension=".zip")
        assert (
            len(sub.results_1d)
            == test_time_series_length * 24 * (len(sub.cross_sections) + 1)
            and len(sub.results_1d.columns) == 6
        )
        assert (
            len(sub.xs_hydrograph) == test_time_series_length * 24
            and len(sub.xs_hydrograph.columns) == 2
        )
        assert (
            len(sub.xs_water_level) == test_time_series_length * 24
            and len(sub.xs_water_level.columns) == 2
        )
        assert (
            len(sub.xs_water_depth) == test_time_series_length * 24
            and len(sub.xs_water_depth.columns) == 2
        )
        sub.read_1d_results(xsid=read_1d_result_xs_id)

        assert read_1d_result_xs_id in sub.xs_hydrograph.columns.tolist()
        assert read_1d_result_xs_id in sub.xs_water_level.columns.tolist()
        assert read_1d_result_xs_id in sub.xs_water_depth.columns.tolist()

    def test_read_chunks(
        self,
        version: int,
        river_cross_section_path: str,
        segment1: int,
        read_1d_result_path: str,
        read_1d_result_xs_id: int,
        test_time_series_length: int,
    ):
        rivers = River("HM", version=version)
        rivers.one_d_result_path = read_1d_result_path
        # rivers.results_paths = {"one_d_result_path": Read1DResult_path}
        rivers.read_xs(river_cross_section_path)
        sub = Reach(segment1, rivers)
        sub.read_1d_results(chunk_size=10000, extension=".zip")
        assert (
            len(sub.results_1d)
            == test_time_series_length * 24 * (len(sub.cross_sections) + 1)
            and len(sub.results_1d.columns) == 6
        )
        assert (
            len(sub.xs_hydrograph) == test_time_series_length * 24
            and len(sub.xs_hydrograph.columns) == 2
        )
        assert (
            len(sub.xs_water_level) == test_time_series_length * 24
            and len(sub.xs_water_level.columns) == 2
        )
        assert (
            len(sub.xs_water_depth) == test_time_series_length * 24
            and len(sub.xs_water_depth.columns) == 2
        )
        sub.read_1d_results(xsid=read_1d_result_xs_id)

        assert read_1d_result_xs_id in sub.xs_hydrograph.columns.tolist()
        assert read_1d_result_xs_id in sub.xs_water_level.columns.tolist()
        assert read_1d_result_xs_id in sub.xs_water_depth.columns.tolist()

    def test_read_zip(
        self,
        version: int,
        river_cross_section_path: str,
        segment1: int,
        read_1d_result_path: str,
        read_1d_result_xs_id: int,
        test_time_series_length: int,
    ):
        rivers = River("HM", version=version)
        rivers.one_d_result_path = read_1d_result_path
        # rivers.results_paths = {"one_d_result_path": Read1DResult_path}
        rivers.read_xs(river_cross_section_path)
        sub = Reach(segment1, rivers)
        sub.read_1d_results(extension=".zip")
        assert (
            len(sub.results_1d)
            == test_time_series_length * 24 * (len(sub.cross_sections) + 1)
            and len(sub.results_1d.columns) == 6
        )
        assert (
            len(sub.xs_hydrograph) == test_time_series_length * 24
            and len(sub.xs_hydrograph.columns) == 2
        )
        assert (
            len(sub.xs_water_level) == test_time_series_length * 24
            and len(sub.xs_water_level.columns) == 2
        )
        assert (
            len(sub.xs_water_depth) == test_time_series_length * 24
            and len(sub.xs_water_depth.columns) == 2
        )
        sub.read_1d_results(xsid=read_1d_result_xs_id)

        assert read_1d_result_xs_id in sub.xs_hydrograph.columns.tolist()
        assert read_1d_result_xs_id in sub.xs_water_level.columns.tolist()
        assert read_1d_result_xs_id in sub.xs_water_depth.columns.tolist()

    def test_read_zip_chunks(
        self,
        version: int,
        river_cross_section_path: str,
        segment1: int,
        read_1d_result_path: str,
        read_1d_result_xs_id: int,
        test_time_series_length: int,
    ):
        rivers = River("HM", version=version)
        rivers.one_d_result_path = read_1d_result_path
        # rivers.results_paths = {"one_d_result_path": Read1DResult_path}
        rivers.read_xs(river_cross_section_path)
        sub = Reach(segment1, rivers)
        sub.read_1d_results(chunk_size=10000, extension=".zip")
        assert (
            len(sub.results_1d)
            == test_time_series_length * 24 * (len(sub.cross_sections) + 1)
            and len(sub.results_1d.columns) == 6
        )
        assert (
            len(sub.xs_hydrograph) == test_time_series_length * 24
            and len(sub.xs_hydrograph.columns) == 2
        )
        assert (
            len(sub.xs_water_level) == test_time_series_length * 24
            and len(sub.xs_water_level.columns) == 2
        )
        assert (
            len(sub.xs_water_depth) == test_time_series_length * 24
            and len(sub.xs_water_depth.columns) == 2
        )
        sub.read_1d_results(xsid=read_1d_result_xs_id)

        assert read_1d_result_xs_id in sub.xs_hydrograph.columns.tolist()
        assert read_1d_result_xs_id in sub.xs_water_level.columns.tolist()
        assert read_1d_result_xs_id in sub.xs_water_depth.columns.tolist()


def test_sub_get_laterals(
    segment1: int,
    version: int,
    river_cross_section_path: str,
    river_network_path: str,
    slope_path: str,
    dates: list,
    interface_bc_path: str,
    interface_bc_folder: str,
    interface_bc_date_format: str,
    interface_laterals_table_path: str,
    interface_laterals_folder: str,
    interface_laterals_date_format: str,
    sub_get_flow_lateral_table,
    create_sub_instance_last_xs: int,
    test_time_series_length: int,
):
    rivers = River("HM", version=version)
    rivers.read_xs(river_cross_section_path)
    rivers.read_river_network(river_network_path)
    rivers.read_slope(slope_path)
    sub = Reach(segment1, rivers)

    interface = Interface("Rhine", start=dates[0])
    interface.readBoundaryConditionsTable(interface_bc_path)
    interface.read_boundary_conditions(
        path=interface_bc_folder, date_format=interface_bc_date_format
    )

    interface.read_xs(river_cross_section_path)
    interface.readLateralsTable(interface_laterals_table_path)
    interface.readLaterals(
        path=interface_laterals_folder, date_format=interface_laterals_date_format
    )

    sub.get_flow(interface)
    sub.get_laterals(create_sub_instance_last_xs)

    assert (
        len(sub.BC) == len(sub.Laterals) == test_time_series_length
        and len(sub.BC.columns) == 1
        and len(sub.Laterals.columns) == len(sub_get_flow_lateral_table) + 1
    )
    assert all(elem in sub.laterals_table for elem in sub_get_flow_lateral_table)


def test_ReadRRMHydrograph_one_location(
    version: int,
    river_cross_section_path: str,
    ReadRRMHydrograph_path: str,
    segment1: int,
    ReadRRMHydrograph_station_id: int,
    ReadRRMHydrograph_date_format: str,
    test_time_series_length: int,
):
    rivers = River("HM", version=version)
    rivers.rrm_path = ReadRRMHydrograph_path
    rivers.read_xs(river_cross_section_path)
    sub = Reach(segment1, rivers)
    sub.read_rrm_hydrograph(
        ReadRRMHydrograph_station_id,
        date_format=ReadRRMHydrograph_date_format,
    )

    assert len(sub.RRM) == test_time_series_length and len(sub.RRM.columns) == 1
    assert sub.RRM.columns.to_list()[0] == ReadRRMHydrograph_station_id


def test_ReadRRMHydrograph_two_location(
    version: int,
    river_cross_section_path: str,
    ReadRRMHydrograph_path: str,
    segment1: int,
    ReadRRMHydrograph_station_id: int,
    ReadRRMHydrograph_date_format: str,
    ReadRRMHydrograph_location_2: int,
    ReadRRMHydrograph_path2: str,
    test_time_series_length: int,
):
    rivers = River("HM", version=version)
    rivers.rrm_path = ReadRRMHydrograph_path
    rivers.read_xs(river_cross_section_path)
    sub = Reach(segment1, rivers)
    sub.read_rrm_hydrograph(
        ReadRRMHydrograph_station_id,
        date_format=ReadRRMHydrograph_date_format,
        location=ReadRRMHydrograph_location_2,
        path2=ReadRRMHydrograph_path2,
    )

    assert len(sub.RRM) == test_time_series_length and len(sub.RRM.columns) == 1
    assert sub.RRM.columns.to_list()[0] == ReadRRMHydrograph_station_id
    assert len(sub.RRM2) == test_time_series_length and len(sub.RRM.columns) == 1
    assert sub.RRM2.columns.to_list()[0] == ReadRRMHydrograph_station_id


def test_ReadUSHydrograph(
    version: int,
    CustomizedRunspath: str,
    river_cross_section_path: str,
    river_network_path: str,
    segment3: int,
    test_time_series_length: int,
    segment3_us_subs: List[int],
):
    rivers = River("HM", version=version)
    rivers.customized_runs_path = CustomizedRunspath
    rivers.read_xs(river_cross_section_path)
    rivers.read_river_network(river_network_path)
    sub = Reach(segment3, rivers)
    sub.read_us_hydrograph()
    assert (
        len(sub.us_hydrographs) == test_time_series_length
        and len(sub.us_hydrographs.columns) == len(segment3_us_subs) + 1
    )
    assert all(elem in sub.us_hydrographs.columns.tolist() for elem in segment3_us_subs)


class TestGetTotalFlow:
    def test_segment_without_bc(
        self,
        version: int,
        river_cross_section_path: str,
        river_network_path: str,
        CustomizedRunspath: str,
        segment3: int,
        segment3_xs: int,
        dates: list,
        interface_bc_path: str,
        interface_bc_folder: str,
        interface_bc_date_format: str,
        interface_laterals_table_path: str,
        interface_laterals_folder: str,
        interface_laterals_date_format: str,
        test_time_series_length: int,
    ):
        """Test_segment_without_bc.

        extract the total flow for a river segment that is in the middle
        of the river and does not have a boundary condition
        """
        rivers = River("HM", version=version)
        rivers.read_xs(river_cross_section_path)
        rivers.read_river_network(river_network_path)
        rivers.customized_runs_path = CustomizedRunspath
        interface = Interface("Rhine", start=dates[0])
        interface.readBoundaryConditionsTable(interface_bc_path)
        interface.read_boundary_conditions(
            path=interface_bc_folder, date_format=interface_bc_date_format
        )

        interface.read_xs(river_cross_section_path)
        interface.readLateralsTable(interface_laterals_table_path)
        interface.readLaterals(
            path=interface_laterals_folder, date_format=interface_laterals_date_format
        )

        sub = Reach(segment3, rivers)
        sub.get_flow(interface)
        sub.read_us_hydrograph()

        sub.get_total_flow(segment3_xs)
        assert len(sub.TotalFlow) == test_time_series_length
        assert "total" in sub.TotalFlow.columns.to_list()

    def test_segment_wit_bc(
        self,
        version: int,
        river_cross_section_path: str,
        river_network_path: str,
        CustomizedRunspath: str,
        segment1: int,
        segment1_xs: int,
        dates: list,
        interface_bc_path: str,
        interface_bc_folder: str,
        interface_bc_date_format: str,
        interface_laterals_table_path: str,
        interface_laterals_folder: str,
        interface_laterals_date_format: str,
        test_time_series_length: int,
    ):
        """Test_segment_without_bc.

        extract the total flow for a river segment that is in the middle
        of the river and does not have a boundary condition
        """
        rivers = River("HM", version=version)
        rivers.read_xs(river_cross_section_path)
        rivers.read_river_network(river_network_path)
        rivers.customized_runs_path = CustomizedRunspath
        interface = Interface("Rhine", start=dates[0])
        interface.readBoundaryConditionsTable(interface_bc_path)
        interface.read_boundary_conditions(
            path=interface_bc_folder, date_format=interface_bc_date_format
        )

        interface.read_xs(river_cross_section_path)
        interface.readLateralsTable(interface_laterals_table_path)
        interface.readLaterals(
            path=interface_laterals_folder, date_format=interface_laterals_date_format
        )

        sub = Reach(segment1, rivers)
        sub.get_flow(interface)
        sub.read_us_hydrograph()

        sub.get_total_flow(segment1_xs)
        assert len(sub.TotalFlow) == test_time_series_length
        assert "total" in sub.TotalFlow.columns.to_list()


def test_PlotQ(
    version: int,
    river_cross_section_path: str,
    river_network_path: str,
    CustomizedRunspath: str,
    read_1d_result_path: str,
    segment3: int,
    create_sub_instance_last_xs: int,
    dates: list,
    interface_bc_path: str,
    interface_bc_folder: str,
    interface_bc_date_format: str,
    interface_laterals_table_path: str,
    interface_laterals_folder: str,
    interface_laterals_date_format: str,
    gauges_table_path: str,
    ReadObservedQ_Path: str,
    nodatavalu: int,
    gauge_date_format: str,
    segment3_specificxs_plot: int,
):
    Calib = rc.Calibration("HM", version=version)
    Calib.readGaugesTable(gauges_table_path)
    Calib.readObservedQ(
        ReadObservedQ_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )

    gauge_i = 0
    gauges = Calib.hm_gauges.loc[Calib.hm_gauges["id"] == segment3, :]
    gauges.index = range(len(gauges))
    station_name = gauges.loc[gauge_i, "oid"]
    gaugename = str(gauges.loc[gauge_i, "name"])
    gaugexs = gauges.loc[gauge_i, "xsid"]
    segment_xs = str(segment3) + "_" + str(gaugexs)

    rivers = River("HM", version=version)
    rivers.read_xs(river_cross_section_path)
    rivers.read_river_network(river_network_path)
    rivers.customized_runs_path = CustomizedRunspath
    rivers.one_d_result_path = read_1d_result_path

    interface = Interface("Rhine", start=dates[0])
    interface.readBoundaryConditionsTable(interface_bc_path)
    interface.read_boundary_conditions(
        path=interface_bc_folder, date_format=interface_bc_date_format
    )

    interface.read_xs(river_cross_section_path)
    interface.readLateralsTable(interface_laterals_table_path)
    interface.readLaterals(
        path=interface_laterals_folder, date_format=interface_laterals_date_format
    )

    sub = Reach(segment3, rivers)
    sub.get_flow(interface)
    sub.read_us_hydrograph()
    sub.read_1d_results(extension=".zip")
    fig, ax = sub.plot_q(
        Calib,
        gaugexs,
        dates[0],
        dates[1],
        station_name,
        gaugename,
        segment_xs,
        specificxs=segment3_specificxs_plot,
        xlabels=5,
        ylabels=5,
    )
    plt.close()
    assert isinstance(fig, Figure)


def test_CalculateQMetrics(
    version: int,
    dates: list,
    river_cross_section_path: str,
    river_network_path: str,
    read_1d_result_path: str,
    gauges_table_path: str,
    ReadObservedQ_Path: str,
    nodatavalu: int,
    gauge_date_format: str,
    segment3: int,
):
    Calib = rc.Calibration("HM", version=version)
    Calib.readGaugesTable(gauges_table_path)
    Calib.readObservedQ(
        ReadObservedQ_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )

    gauge_i = 0
    gauges = Calib.hm_gauges.loc[Calib.hm_gauges["id"] == segment3, :]
    gauges.index = range(len(gauges))
    station_name = gauges.loc[gauge_i, "oid"]

    rivers = River("HM", version=version, start=dates[0])
    rivers.one_d_result_path = read_1d_result_path
    rivers.read_xs(river_cross_section_path)
    rivers.read_river_network(river_network_path)

    sub = Reach(segment3, rivers)
    sub.read_1d_results(extension=".zip")

    # without filter
    sub.calculate_q_metrics(Calib, station_name, sub.last_xs)

    sub.calculate_q_metrics(
        Calib, station_name, sub.last_xs, Filter=True, start=dates[0], end=dates[1]
    )


def test_PlotHydrographProgression(
    version: int,
    dates: list,
    river_cross_section_path: str,
    river_network_path: str,
    read_1d_result_path: str,
    segment3: int,
):
    rivers = River("HM", version=version, start=dates[0])
    rivers.one_d_result_path = read_1d_result_path
    rivers.read_xs(river_cross_section_path)
    rivers.read_river_network(river_network_path)

    sub = Reach(segment3, rivers)
    sub.read_1d_results(extension=".zip")

    xss = []
    start = dates[0]
    end = dates[1]
    fig, ax = sub.plot_hydrograph_progression(
        xss,
        start,
        end,
        from_xs=None,
        to_xs=None,
        line_width=2,
        spacing=20,
        fig_size=(6, 4),
        xlabels=5,
    )
    plt.close()
    assert isinstance(fig, Figure)


def test_PlotWL(
    version: int,
    dates: list,
    gauges_table_path: str,
    ReadObservedWL_Path: str,
    nodatavalu: int,
    gauge_date_format: str,
    segment3: int,
    river_cross_section_path: str,
    river_network_path: str,
    read_1d_result_path: str,
):
    Calib = rc.Calibration("HM", version=version)
    Calib.readGaugesTable(gauges_table_path)
    Calib.readObservedWL(
        ReadObservedWL_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )

    gauge_i = 0
    gauges = Calib.hm_gauges.loc[Calib.hm_gauges["id"] == segment3, :]
    gauges.index = range(len(gauges))
    station_name = gauges.loc[gauge_i, "oid"]
    gaugename = str(gauges.loc[gauge_i, "name"])
    gaugexs = gauges.loc[gauge_i, "xsid"]

    rivers = River("HM", version=version)
    rivers.read_xs(river_cross_section_path)
    rivers.read_river_network(river_network_path)
    rivers.one_d_result_path = read_1d_result_path

    sub = Reach(segment3, rivers)
    sub.read_1d_results(extension=".zip")

    sub.plot_wl(
        Calib, dates[0], dates[1], gaugexs, station_name, gaugename, plotgauge=True
    )
    plt.close()


def test_CalculateWLMetrics(
    version: int,
    dates: list,
    river_cross_section_path: str,
    river_network_path: str,
    read_1d_result_path: str,
    gauges_table_path: str,
    ReadObservedWL_Path: str,
    nodatavalu: int,
    gauge_date_format: str,
    segment3: int,
):
    Calib = rc.Calibration("HM", version=version)
    Calib.readGaugesTable(gauges_table_path)
    Calib.readObservedWL(
        ReadObservedWL_Path,
        dates[0],
        dates[1],
        nodatavalu,
        gauge_date_format=gauge_date_format,
    )

    gauge_i = 0
    gauges = Calib.hm_gauges.loc[Calib.hm_gauges["id"] == segment3, :]
    gauges.index = range(len(gauges))
    station_name = gauges.loc[gauge_i, "oid"]

    rivers = River("HM", version=version, start=dates[0])
    rivers.one_d_result_path = read_1d_result_path
    rivers.read_xs(river_cross_section_path)
    rivers.read_river_network(river_network_path)

    sub = Reach(segment3, rivers)
    sub.read_1d_results(extension=".zip")

    # without filter
    sub.calculate_wl_metrics(Calib, station_name, sub.last_xs)

    sub.calculate_wl_metrics(
        Calib, station_name, sub.last_xs, Filter=True, start=dates[0], end=dates[1]
    )


def test_SaveHydrograph(
    version: int,
    river_cross_section_path: str,
    river_network_path: str,
    CustomizedRunspath: str,
    read_1d_result_path: str,
    segment3: int,
):
    rivers = River("HM", version=version)
    rivers.read_xs(river_cross_section_path)
    rivers.read_river_network(river_network_path)
    rivers.customized_runs_path = CustomizedRunspath
    rivers.one_d_result_path = read_1d_result_path
    sub = Reach(segment3, rivers)
    sub.read_1d_results(extension=".zip")
    sub.save_hydrograph(sub.last_xs)
    # option 2
    sub.save_hydrograph(sub.last_xs, Option=2)


def test_ReadBoundaryConditions(
    version: int,
    river_cross_section_path: str,
    river_network_path: str,
    dates: list,
    onemin_days: int,
    read_1d_result_path: str,
    usbc_path: str,
    segment3: int,
    test_time_series_length: int,
    test_hours: list,
):
    rivers = River("HM", version=version, start=dates[0])
    rivers.read_xs(river_cross_section_path)
    rivers.read_river_network(river_network_path)
    rivers.one_d_result_path = read_1d_result_path
    rivers.us_bc_path = usbc_path
    sub = Reach(segment3, rivers)
    sub.read_1d_results(extension=".zip")
    # read only 10 days

    days = int(dates[0].split("-")[-1]) + onemin_days
    day2 = f"{dates[0][:-2]}{days}"

    sub.read_boundary_conditions(start=dates[0], end=day2)

    assert len(sub.QBC) == onemin_days + 1 and all(
        elem in test_hours for elem in sub.QBC.columns.tolist()
    )
    assert len(sub.HBC) == onemin_days + 1 and all(
        elem in test_hours for elem in sub.HBC.columns.tolist()
    )


def test_ReadSubDailyResults(
    version: int,
    river_cross_section_path: str,
    river_network_path: str,
    dates: list,
    segment3: int,
    lastsegment: bool,
    subdailyresults_path: str,
    usbc_path: str,
    segment3_xs_ids_list: list,
    subdaily_no_timesteps: int,
    onemin_results_dates: list,
    onemin_results_len: int,
):
    rivers = River("HM", version=version, start=dates[0])
    rivers.read_xs(river_cross_section_path)
    rivers.read_river_network(river_network_path)
    rivers.us_bc_path = usbc_path
    rivers.one_min_result_path = subdailyresults_path
    sub = Reach(segment3, rivers)
    sub.read_sub_daily_results(
        onemin_results_dates[0], onemin_results_dates[1], last_river_reach=lastsegment
    )
    assert len(sub.h) == onemin_results_len * subdaily_no_timesteps
    assert all(elem in sub.h.columns.tolist() for elem in segment3_xs_ids_list)
    assert len(sub.q) == onemin_results_len * subdaily_no_timesteps
    assert all(elem in sub.q.columns.tolist() for elem in segment3_xs_ids_list)
    assert (
        len(sub.q_bc_1min.columns) == subdaily_no_timesteps
        and len(sub.q_bc_1min) == onemin_results_len
    )
    assert (
        len(sub.h_bc_1min.columns) == subdaily_no_timesteps
        and len(sub.h_bc_1min) == onemin_results_len
    )


def test_PlotBC(
    version: int,
    river_cross_section_path: str,
    river_network_path: str,
    dates: list,
    segment3: int,
    lastsegment: bool,
    subdailyresults_path: str,
    usbc_path: str,
    onemin_results_dates: list,
):
    rivers = River("HM", version=version, start=dates[0])
    rivers.read_xs(river_cross_section_path)
    rivers.read_river_network(river_network_path)
    rivers.us_bc_path = usbc_path
    rivers.one_min_result_path = subdailyresults_path
    sub = Reach(segment3, rivers)
    sub.read_sub_daily_results(
        onemin_results_dates[0], onemin_results_dates[1], last_river_reach=lastsegment
    )
    sub.plot_bc(dates[0])


def test_StatisticalProperties(
    version: int,
    distribution_properties_fpath: str,
    statistical_properties_columns: list,
    distributionpr_gev_columns: list,
    distributionpr_gum_columns: list,
):
    rivers = River("HM", version=version)
    rivers.statistical_properties(distribution_properties_fpath, distibution="GEV")
    assert all(
        elem in rivers.SP.columns.to_list() for elem in statistical_properties_columns
    )
    assert all(
        elem in rivers.SP.columns.to_list() for elem in distributionpr_gev_columns[:3]
    )

    rivers.statistical_properties(distribution_properties_fpath, distibution="Gumbel")
    assert all(
        elem in rivers.SP.columns.to_list() for elem in statistical_properties_columns
    )
    assert all(
        elem in rivers.SP.columns.to_list() for elem in distributionpr_gum_columns[:2]
    )


def test_GetCapacity(
    version: int,
    river_cross_section_path: str,
    distribution_properties_hm_results_fpath: str,
):
    rivers = River("HM", version=version)
    rivers.read_xs(river_cross_section_path)
    rivers.statistical_properties(distribution_properties_hm_results_fpath)
    rivers.get_river_capacity("Qbkf")
    rivers.get_river_capacity("Qc2", option=2)
    cols = rivers.cross_sections.columns.tolist()
    assert "Slope" in cols
    assert "Qbkf" in cols
    assert "QbkfRP" in cols
    assert "Qc2" in cols
    assert "Qc2RP" in cols


def test_collect_1d_results(
    combine_rdir: str,
    combine_save_to: str,
    separated_folders: List[str],
    separated_folders_file_names: List[str],
):
    days_in_combined_file = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    days_should_be_in_left_file = [1, 7]
    days_should_be_in_right_file = [8]
    left = True
    right = True
    one_d = True
    filter_by_name = True
    River.collect_1d_results(
        combine_rdir,
        separated_folders,
        left,
        right,
        combine_save_to,
        one_d,
        filter_by_name=filter_by_name,
    )
    combined_files = os.listdir(combine_save_to)
    assert len(combined_files) - 1 == len(separated_folders_file_names)

    # check the results of the 1.txt files
    reach_result_file = separated_folders_file_names[0]
    df = pd.read_csv(
        f"{combine_save_to}/{reach_result_file}", delimiter=r"\s", header=None
    )
    assert len(df) > 0, f"the {reach_result_file} should have results not empty"
    days_not_in_combined = set(days_in_combined_file) - set(df[0].unique())
    assert len(days_not_in_combined) == 0, (
        f"the {reach_result_file} file should have results for 10 days but there "
        f"are missing days: {days_not_in_combined}"
    )
    os.remove(f"{combine_save_to}/{reach_result_file}")
    # check the results of the 1_left.txt files
    reach_result_file = separated_folders_file_names[1]
    df = pd.read_csv(
        f"{combine_save_to}/{reach_result_file}", delimiter=r"\s", header=None
    )
    days_not_in_combined = set(days_should_be_in_left_file) - set(df[0].unique())
    assert len(days_not_in_combined) == 0, (
        f"the {reach_result_file} file should have results for 10 days but there "
        f"are missing days: {days_not_in_combined}"
    )
    os.remove(f"{combine_save_to}/{reach_result_file}")
    # check the results of the 1_right.txt files
    reach_result_file = separated_folders_file_names[2]
    df = pd.read_csv(
        f"{combine_save_to}/{reach_result_file}", delimiter=r"\s", header=None
    )
    days_not_in_combined = set(days_should_be_in_right_file) - set(df[0].unique())
    assert len(days_not_in_combined) == 0, (
        f"the {reach_result_file} file should have results for 10 days but there "
        f"are missing days: {days_not_in_combined}"
    )
    os.remove(f"{combine_save_to}/{reach_result_file}")


def test_get_bankfull_depth(
    version: int, river_cross_section_path: str, xs_total_no: int, xs_col_no: int
):
    rivers = River("HM", version=version)
    rivers.read_xs(river_cross_section_path)

    def bankfulldepth(b):
        return 0.6354 * (b / 0.7093) ** 0.3961

    # recalculate the original derived depth from the hyudraulic geomerty relations
    rivers.get_bankfull_depth(bankfulldepth, "dbf2")
    xs = rivers.cross_sections
    assert "dbf2" in xs.columns


def test_read_overtopping(
    version: int,
    river_cross_section_path: str,
    xs_total_no: int,
    xs_col_no: int,
    overtopping_files_dir: str,
):
    rivers = River("HM", version=version)
    rivers.read_xs(river_cross_section_path)

    rivers.parse_overtopping(
        overtopping_result_path=overtopping_files_dir, delimiter=","
    )
    assert hasattr(rivers, "overtopping_reaches_left")
    assert hasattr(rivers, "overtopping_reaches_right")
    assert len(rivers.overtopping_reaches_left.keys()) == 1
    assert len(rivers.overtopping_reaches_right.keys()) == 1


def test_get_event_start_end(
    version: int,
    river_cross_section_path: str,
    xs_total_no: int,
    xs_col_no: int,
    overtopping_files_dir: str,
    event_index_file: str,
):
    rivers = River("HM", version=version)
    rivers.read_xs(river_cross_section_path)
    event = Event.read_event_index("test", event_index_file, start="1955-01-01")
    max_overtopping_ind = event.event_index["cells"].idxmax()

    _, start_day = event.get_event_start(max_overtopping_ind)
    _, end_day = event.get_event_end(max_overtopping_ind)
    assert start_day == 8195
    assert end_day == 8200


class TestEvent:
    def test_get_overtopped_xs_one_day(
        self,
        version: int,
        river_cross_section_path: str,
        xs_total_no: int,
        xs_col_no: int,
        overtopping_files_dir: str,
        event_index_file: str,
    ):
        rivers = River("HM", version=version)
        rivers.read_xs(river_cross_section_path)
        rivers.parse_overtopping(
            overtopping_result_path=overtopping_files_dir, delimiter=","
        )
        event = Event.read_event_index("test", event_index_file, start="1955-01-01")
        rivers.event_index = event.event_index
        # _, start_day = event.get_event_start(0)
        # _, end_day = event.get_event_end(0)
        event_details = event.get_event_by_index(1)
        start_day = event_details["start"]
        xs_flood_l, xs_flood_r = rivers.get_overtopped_xs(start_day, False)
        assert xs_flood_l == [1]
        assert xs_flood_r == [1, 5, 7, 9, 12, 15]

    def test_get_overtopped_xs_all_days(
        self,
        version: int,
        river_cross_section_path: str,
        xs_total_no: int,
        xs_col_no: int,
        overtopping_files_dir: str,
        event_index_file: str,
    ):
        rivers = River("HM", version=version)
        rivers.read_xs(river_cross_section_path)
        rivers.parse_overtopping(
            overtopping_result_path=overtopping_files_dir, delimiter=","
        )
        event = Event.read_event_index("test", event_index_file, start="1955-01-01")
        rivers.event_index = event.event_index

        event_details = event.get_event_by_index(1)
        end_day = event_details["end"]

        xs_flood_l, xs_flood_r = rivers.get_overtopped_xs(end_day, True)
        assert xs_flood_l == [8, 1, 10, 5]
        assert xs_flood_r == [1, 5, 7, 9, 12, 15]


def test_get_flooded_reaches(
    version: int,
    river_cross_section_path: str,
    xs_total_no: int,
    xs_col_no: int,
    overtopping_files_dir: str,
    event_index_file: str,
):
    rivers = River("HM", version=version)
    rivers.read_xs(river_cross_section_path)
    rivers.parse_overtopping(
        overtopping_result_path=overtopping_files_dir, delimiter=","
    )
    event = Event.read_event_index("test", event_index_file, start="1955-01-01")
    rivers.event_index = event.event_index
    # _, start_day = event.get_event_start(0)

    event_details = event.get_event_by_index(1)
    start_day = event_details["start"]

    xs_flood_l, xs_flood_r = rivers.get_overtopped_xs(start_day, True)
    flooded_reaches = rivers.get_flooded_reaches(overtopped_xs=xs_flood_l + xs_flood_r)
    assert flooded_reaches == [1]


def test_river_detailed_overtopping(
    version: int,
    river_cross_section_path: str,
    xs_total_no: int,
    xs_col_no: int,
    overtopping_files_dir: str,
    event_index_file: str,
):
    rivers = River("HM", version=version)
    rivers.one_d_result_path = overtopping_files_dir
    rivers.read_xs(river_cross_section_path)
    rivers.parse_overtopping(
        overtopping_result_path=overtopping_files_dir, delimiter=","
    )
    event = Event.read_event_index("test", event_index_file, start="1955-01-01")
    rivers.event_index = event.event_index
    _, start_day = event.get_event_start(0)
    _, end_day = event.get_event_end(0)
    xs_flood_l, xs_flood_r = rivers.get_overtopped_xs(start_day, True)
    flooded_reaches = rivers.get_flooded_reaches(overtopped_xs=xs_flood_l + xs_flood_r)

    event_days = list(range(start_day, end_day + 1))
    rivers.detailed_overtopping(flooded_reaches, event_days, delimiter=",")
    assert isinstance(rivers.detailed_overtopping_left, DataFrame)
    assert isinstance(rivers.detailed_overtopping_right, DataFrame)
    assert "sum" in rivers.detailed_overtopping_right.columns
    assert "sum" in rivers.detailed_overtopping_left.columns


def test_get_days(
    version: int,
    river_cross_section_path: str,
    xs_total_no: int,
    xs_col_no: int,
    overtopping_files_dir: str,
    event_index_file: str,
):
    rivers = River("HM", version=version)
    rivers.read_xs(river_cross_section_path)
    rivers.one_d_result_path = overtopping_files_dir
    reach_id = 1
    reach = Reach(reach_id, rivers)
    alter1, alter2 = reach.get_days(35, 25, delimiter=",")
    assert alter1 == 35
    assert alter2 == 30


def test_reach_detailed_overtopping(
    version: int,
    river_cross_section_path: str,
    xs_total_no: int,
    xs_col_no: int,
    overtopping_files_dir: str,
    event_index_file: str,
):
    rivers = River("HM", version=version)
    rivers.read_xs(river_cross_section_path)
    rivers.one_d_result_path = overtopping_files_dir
    reach_id = 1
    reach = Reach(reach_id, rivers)
    reach.read_1d_results(fill_missing=False, delimiter=",")
    event_days = [35, 36, 37, 38]
    reach.detailed_overtopping(event_days, delimiter=",")

    assert isinstance(reach.all_overtopping_vs_xs, DataFrame)
    assert reach.all_overtopping_vs_xs.index.tolist() == [1, 5, 7, 8, 9, 10, 12, 15]
    assert reach.all_overtopping_vs_xs.columns.tolist() == ["sum"]

    assert isinstance(reach.all_overtopping_vs_time, DataFrame)
    assert reach.all_overtopping_vs_time.loc[:, "id"].tolist() == event_days
    assert reach.all_overtopping_vs_time.columns.tolist() == [
        "id",
        "Overtopping",
        "date",
    ]

    cols = [f"reach-{reach.id}"] + reach.xs_names + ["sum"]
    rows = event_days + ["sum"]
    assert isinstance(reach.detailed_overtopping_right, DataFrame)
    assert isinstance(reach.detailed_overtopping_left, DataFrame)
    assert reach.detailed_overtopping_right.columns.tolist() == cols
    assert reach.detailed_overtopping_left.columns.tolist() == cols
    assert reach.detailed_overtopping_right.index.tolist() == rows
    assert reach.detailed_overtopping_left.index.tolist() == rows
