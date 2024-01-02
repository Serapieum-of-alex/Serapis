import pytest

from tests.calibration.conftest import *
from tests.inputs.conftest import *
from tests.interface.conftest import *
from tests.river.conftest import *

time_series_length = 80
hours = list(range(1, 25))


@pytest.fixture(scope="module")
def version() -> int:
    return 3


@pytest.fixture(scope="module")
def dates() -> list:
    start = "1955-01-01"
    end = "1955-03-21"
    return [start, end]


@pytest.fixture(scope="module")
def rrm_start() -> str:
    return "1955-1-1"


@pytest.fixture(scope="module")
def nodatavalu() -> int:
    return -9


@pytest.fixture(scope="module")
def xs_total_no() -> int:
    return 300


@pytest.fixture(scope="module")
def xs_col_no() -> int:
    return 17


@pytest.fixture(scope="module")
def test_time_series_length() -> int:
    return time_series_length


@pytest.fixture(scope="module")
def test_hours() -> list:
    return hours


@pytest.fixture(scope="module")
def combine_rdir() -> str:
    return "tests/data/results/combin_results"


@pytest.fixture(scope="module")
def combine_save_to() -> str:
    return "tests/data/results/combin_results/combined"


@pytest.fixture(scope="module")
def separated_folders() -> List[str]:
    return ["1d(1-5)", "1d(6-10)"]


@pytest.fixture(scope="module")
def separated_folders_file_names() -> List[str]:
    return ["1.txt", "1_left.txt", "1_right.txt"]


@pytest.fixture(scope="module")
def overtopping_file() -> str:
    return "tests/data/overtopping.txt"


@pytest.fixture(scope="module")
def volume_error_file() -> str:
    return "tests/data/VolError.txt"


@pytest.fixture(scope="module")
def event_instance_attrs() -> List[str]:
    return [
        "_left_overtopping_suffix",
        "_right_overtopping_suffix",
        "_duration_prefix",
        "_return_period_prefix",
        "_two_d_result_path",
        "compressed",
        "extracted_values",
        "_event_index",
    ]


@pytest.fixture(scope="module")
def event_index_volume_attrs() -> List[str]:
    return ["DEMError", "StepError", "TooMuchWater", "VolError", "VolError2"]


@pytest.fixture(scope="module")
def event_index_volume_attrs2() -> List[str]:
    return ["id", "date", "continue", "ind_diff", "duration", "cells"]
