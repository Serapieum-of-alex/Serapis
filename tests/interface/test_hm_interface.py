from serapis.interface import Interface


def test_create_interface_instance(dates: list):
    Interface("Rhine", start=dates[0])


def test_readLateralsTable(
    dates: list,
    river_cross_section_path: str,
    interface_laterals_table_path: str,
):
    IF = Interface("Rhine", start=dates[0])
    IF.read_xs(river_cross_section_path)
    IF.read_laterals_table(interface_laterals_table_path)

    assert len(IF.laterals_table) == 9 and len(IF.laterals_table.columns) == 2


class TestreadLaterals:
    def test_without_parallel_io(
        self,
        dates: list,
        river_cross_section_path: str,
        interface_laterals_table_path: str,
        interface_laterals_folder: str,
        interface_laterals_date_format: str,
        test_time_series_length: int,
    ):
        IF = Interface("Rhine", start=dates[0])
        IF.read_xs(river_cross_section_path)
        IF.read_laterals_table(interface_laterals_table_path)
        IF.read_laterals(
            path=interface_laterals_folder, date_format=interface_laterals_date_format
        )
        assert (
            len(IF.Laterals) == test_time_series_length
            and len(IF.Laterals.columns) == 10
        )

    def test_with_parallel_io(
        self,
        dates: list,
        river_cross_section_path: str,
        interface_laterals_table_path: str,
        interface_laterals_folder: str,
        interface_laterals_date_format: str,
        test_time_series_length: int,
    ):
        IF = Interface("Rhine", start=dates[0])
        IF.read_xs(river_cross_section_path)
        IF.read_laterals_table(interface_laterals_table_path)
        IF.read_laterals(
            path=interface_laterals_folder,
            date_format=interface_laterals_date_format,
            cores=True,
        )
        assert (
            len(IF.Laterals) == test_time_series_length
            and len(IF.Laterals.columns) == 10
        )


def test_readBoundaryConditionsTable(
    dates: list,
    interface_bc_path: str,
):
    IF = Interface("Rhine", start=dates[0])
    IF.read_boundary_conditions_table(interface_bc_path)

    assert len(IF.bc_table) == 2 and len(IF.bc_table.columns) == 2


def test_ReadBoundaryConditions(
    dates: list,
    interface_bc_path: str,
    interface_bc_folder: str,
    interface_bc_date_format: str,
    test_time_series_length: int,
):
    IF = Interface("Rhine", start=dates[0])
    IF.read_boundary_conditions_table(interface_bc_path)
    IF.read_boundary_conditions(
        path=interface_bc_folder, date_format=interface_bc_date_format
    )

    assert len(IF.BC) == test_time_series_length and len(IF.BC.columns) == 3


def test_ReadRRMProgression(
    dates: list,
    river_cross_section_path: str,
    interface_laterals_table_path: str,
    rrm_resutls_hm_location: str,
    interface_laterals_date_format: str,
    laterals_number_ts: int,
    no_laterals: int,
):
    IF = Interface("Rhine", start=dates[0])
    IF.read_xs(river_cross_section_path)
    IF.read_laterals_table(interface_laterals_table_path)
    IF.read_laterals(
        path=rrm_resutls_hm_location,
        date_format=interface_laterals_date_format,
        laterals=False,
    )
    assert len(IF.routed_rrm) == laterals_number_ts
    # number of laterals + the total column
    assert len(IF.routed_rrm.columns) == no_laterals + 1
