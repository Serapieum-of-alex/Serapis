"""Inputs."""
import datetime as dt
import os
import zipfile
from typing import Union
import numpy as np
import pandas as pd
from loguru import logger
from osgeo import gdal
from scipy.stats import gumbel_r
from statista.eva import ams_analysis
from serapis.river import River


class Inputs(River):
    """Hydraulic model Inputs class.

    Methods
    -------
        1- ExtractHydrologicalInputs
        2- StatisticalProperties
        3- WriteHQFile
        4- ReturnPeriod
        5- ReadRIMResult
        6- CreateTraceALL
    """

    def __init__(self, name, version: int = 3):
        """Input.

            Inputs is an object to create the inputs for the river model

        Parameters
        ----------
        name: [str]

        version: [int]
            model version
        """
        self.Name = name
        self.version = version
        self.statistical_properties = None
        self.distribution_properties = None

    def extract_hydrological_inputs(
        self, weather_generator, file_prefix, realization, path, locations, save_path
    ):
        """extract_hydrological_inputs.

        Parameters
        ----------
        weather_generator : TYPE
            DESCRIPTION.
        file_prefix : TYPE
            DESCRIPTION.
        realization : [Integer]
            type the number of the realization (the order of the 1000-year run by rainfall runoff model).
        path : [String]
             rrm_result_file is the naming format you used in naming the result files of the discharge values stored
             with the name of the file as out+realization number + .dat (ex out15.dat).
        locations : [String]
            text file containing the list of sub-basins IDs or computational nodes ID you have used to run the rrm
            and store the results.
        save_path : [String]
            path to the folder where you want to save the separate file for each sub-basin.

        Returns
        -------
        None.
        """
        if weather_generator:
            # weather_generator
            rrm_result_file = file_prefix + str(realization) + ".dat"
            # 4-5
            # check whether the name of the realization the same as the name of 3 the saving file or not to prevent
            # any confusion in saving the files
            if int(realization) <= 9:
                assert int(rrm_result_file[-5:-4]) == int(
                    save_path[-1]
                ), " Wrong files sync "
            else:
                assert int(rrm_result_file[-6:-4]) == int(
                    save_path[-2:]
                ), " Wrong files sync "
        else:
            # Observed data.
            rrm_result_file = file_prefix + str(realization) + ".dat"

        # rrm writes the year as the first column then day as a second column and the discharge values start from the
        # thirst column, so you have to write number of columns to be ignored at the beginning

        ignore_columns = 2

        # read SWIM result file
        swim_data = pd.read_csv(
            f"{path}/{rrm_result_file}", delimiter=r"\s+", header=None
        )
        nodes = pd.read_csv(f"{path}/{locations}", header=None)

        for i in range(len(nodes)):
            swim_data.loc[:, i + ignore_columns].to_csv(
                f"{save_path}/{nodes.loc[i, 0]}.txt", header=None, index=None
            )

    def get_statistical_properties(
        self,
        gauges: list,
        rdir: str,
        start: str,
        warm_up_period: int,
        save_plots: bool,
        save_to: str,
        filter_out: Union[bool, float, int] = False,
        distribution: str = "GEV",
        method: str = "lmoments",
        quartile: float = 0,
        significance_level: float = 0.1,
        file_extension: str = ".txt",
        date_format: str = "%Y-%m-%d",
    ):
        """get_statistical_properties.

            get_statistical_properties method reads the discharge hydrographs of a rainfall runoff model and
            hydraulic model for some computational nodes and calculates some statistical properties.

            the code assumes that the time series are of a daily temporal resolution, and that the hydrological year
            is 1-Nov/31-Oct (Petrow and Merz, 2009, JoH).

        Parameters
        ----------
        gauges : [list]
            The list which contains the ID of the gauges you want to do the statistical analysis for, the ObservedFile
            should contain the discharge time series of these nodes in order.
        rdir : [str]
            The directory where the time series files exist.
        start : [string]
            the beginning date of the time series.
        warm_up_period : [int]
            The number of days you want to neglect at the beginning of the simulation (warm up period).
        save_plots : [Bool]
            True if you want to save the plots.
        save_to : [str]
            the rdir where you want to save the statistical properties.
        filter_out: [Bool]
            For observed or hydraulic model data it has gaps at times, where the model did not run or gaps in the
            observed data if these gap days are filled with a specific value, and you want to ignore it here give
            filter_out = Value you want.
        distribution: [str]
            Default is "GEV".
        method: [str]
            available methods are 'mle', 'mm', 'lmoments', optimization. Default is "lmoments".
        quartile: [float]
            Default is 0.
        significance_level:
            Default is [0.1].
        file_extension: [str]
            Default is '.txt'.
        date_format: [str]
            Default is "%Y-%m-%d".

        Returns
        -------
        statistical-properties.csv:
            file containing some statistical properties like mean, std, min, 5%, 25%, median, 75%, 95%, max, t_beg,
            t_end, nyr, q1.5, q2, q5, q10, q25, q50, q100, q200, q500.
        """
        if not isinstance(gauges, list):
            raise TypeError("gauges should be either a rdir or a list")

        # hydrographs
        time_series = pd.DataFrame()
        # for the hydraulic model results
        logger.info(
            "The function ignores the date column in the time series files and starts from the given start"
            "parameter to the function so check if it is the same start date as in the files"
        )

        skip = []
        for gauge_i in gauges:
            try:
                time_series.loc[:, int(gauge_i)] = pd.read_csv(
                    f"{rdir}/{int(gauge_i)}{file_extension}",
                    skiprows=1,
                    header=None,
                )[1].tolist()
            except FileNotFoundError:
                logger.warning(f"File {int(gauge_i)}{file_extension} does not exist")
                skip.append(int(gauge_i))

        start_date = dt.datetime.strptime(start, date_format)
        end_date = start_date + dt.timedelta(days=time_series.shape[0] - 1)
        ind = pd.date_range(start_date, end_date)
        time_series.index = ind

        # neglect the first year (warmup year) in the time series
        time_series = time_series.loc[
            start_date + dt.timedelta(days=warm_up_period) : end_date, :
        ]

        statistical_properties, distribution_properties = ams_analysis(
            time_series,
            ams_start="A-OCT",
            save_plots=save_plots,
            save_to=save_to,
            filter_out=filter_out,
            distribution=distribution,
            method=method,
            # estimate_parameters=estimate_parameters,
            quartile=quartile,
            significance_level=significance_level,
        )

        # Output file
        statistical_properties.to_csv(
            f"{save_to}/statistical-properties.csv", float_format="%.4f"
        )

        distribution_properties.to_csv(
            f"{save_to}/DistributionProperties.csv", float_format="%.4f"
        )

        self.statistical_properties = statistical_properties
        self.distribution_properties = distribution_properties

    @staticmethod
    def StringSpace(Inp):
        """StringSpace."""
        return str(Inp) + "  "

    def return_period(
        self,
        maps_path,
        prefix,
        distribution_properties_file,
        trace_file,
        sub_basin_file,
        replacement_file,
        hydrological_inputs_path,
        sub_id_map,
        extra_subs_file,
        from_file,
        to_file,
        save_to,
        wpath,
    ):
        """Return period."""
        all_results = os.listdir(maps_path)
        # list of the Max Depth files only
        max_depth_list = list()
        for i in range(len(all_results)):
            if all_results[i].startswith(prefix):
                max_depth_list.append(all_results[i])
        # Read Inputs
        # read the Distribution parameters for each upstream computational node
        distribution_properties = pd.read_csv(distribution_properties_file)
        us_node = pd.read_csv(trace_file, header=None)
        us_node.columns = ["SubID", "US", "DS"]
        # get the sub-basin id from the guide file it is the same shape in RIM1.0 and RIM2.0
        subs_id = pd.read_csv(sub_basin_file, header=None, usecols=[0])

        replacement_sub = pd.read_csv(replacement_file)

        # read the hydrograph for all the US nodes
        # start_date = "1950-1-1"
        # start_date = dt.datetime.strptime(start_date,"%Y-%m-%d")
        # ind = pd.date_range(start_date, StartDate + dt.timedelta(days = NoYears*365), freq = "D")

        ind = range(
            1,
            len(
                pd.read_csv(
                    hydrological_inputs_path
                    + "/"
                    + str(int(us_node.loc[subs_id.loc[10, 0] - 1, "US"]))
                    + ".txt"
                ).values
            ),
        )

        hydrographs = pd.DataFrame(index=ind, columns=subs_id[0].to_list())

        for i in range(len(subs_id)):
            #    i=1
            # search for the SubId in the us_node, or it is listed by order, so subID=343 exist
            # in the row 342 (SubID-1)
            # np.where(us_node['SubID'] == subs_id.loc[i,0])
            try:
                if int(us_node.loc[subs_id.loc[i, 0] - 1, "US"]) != -1:
                    hydrographs.loc[:, subs_id.loc[i, 0]] = pd.read_csv(
                        hydrological_inputs_path
                        + "/"
                        + str(int(us_node.loc[subs_id.loc[i, 0] - 1, "US"]))
                        + ".txt"
                    ).values[: len(hydrographs)]
            except:
                other_sub_loc = np.where(
                    replacement_sub["missing"] == subs_id.loc[i, 0]
                )[0][0]
                if (
                    int(
                        us_node.loc[
                            replacement_sub.loc[other_sub_loc, "replacement"] - 1, "US"
                        ]
                    )
                    != -1
                ):
                    hydrographs.loc[:, subs_id.loc[i, 0]] = pd.read_csv(
                        hydrological_inputs_path
                        + "/"
                        + str(
                            int(
                                us_node.loc[
                                    replacement_sub.loc[other_sub_loc, "replacement"]
                                    - 1,
                                    "US",
                                ]
                            )
                        )
                        + ".txt"
                    ).values[: len(hydrographs)]

        # read sub basin map id
        sub_id_map = gdal.Open(sub_id_map)
        sub_id_map_v = sub_id_map.ReadAsArray()

        # read the added subs' reference text file
        extra_subs = pd.read_csv(extra_subs_file)

        # function to write the numbers in the ASCII file

        # read Max depth map
        check = list()
        klist = list()

        if to_file == "end" or to_file > len(max_depth_list):
            to_file = len(max_depth_list)

        # from_file = 48
        # to_file = from_file +1

        for k in range(from_file, to_file):
            try:
                # open the zip file
                compressed_file = zipfile.ZipFile(f"{maps_path}/{max_depth_list[k]}")
            except:
                print("Error Opening the compressed file")
                check.append(max_depth_list[k][len(prefix) : -4])
                klist.append(k)
                continue

            # get the file name
            file_name = compressed_file.infolist()[0]
            # get the time step from the file name
            timestep = int(file_name.filename[len(prefix) : -4])
            print("File= " + str(timestep))

            asci_file = compressed_file.open(file_name)
            f = asci_file.readlines()
            spatial_ref = f[:6]
            ascii_raw = f[6:]
            # asci_file = compressed_file.open(file_name)
            # ascii_raw = asci_file.readlines()[6:]
            rows = len(ascii_raw)
            cols = len(ascii_raw[0].split())
            max_depth = np.ones((rows, cols), dtype=np.float32)
            # read the ascii file
            for i in range(rows):
                x = ascii_raw[i].split()
                max_depth[i, :] = list(map(float, x))

            # check on the values of the water depth
            #    if np.shape(max_depth[np.isnan(max_depth)])[0] > 0:
            #        check.append(timestep)
            #        print("Error Check Max Depth values")
            #        continue

            # plotting to check values
            #    fromrow = np.where(max_depth == max_depth.max())[0][0]
            #    fromcol = np.where(max_depth == max_depth.max())[1][0]
            #    plt.imshow(max_depth[fromrow-20:fromrow+20,fromcol-20:fromcol+20])
            #    plt.imshow(max_depth)
            #    plt.colorbar()

            # get the Peak of the hydrograph for the whole event
            # (14 days before the end of the event)
            max_values_file = hydrographs.loc[timestep - 14 : timestep, :]
            max_values = max_values_file.max().values.tolist()
            return_period = list()

            # Calculate the Return period for the max Q at this time step for each
            for i in range(len(max_values)):
                # if the sub-basin is a lateral and not routed in RIM, it will not have a
                # hydrograph
                if np.isnan(max_values[i]):
                    return_period.append(np.nan)
                if not np.isnan(max_values[i]):
                    # np.where(us_node['SubID'] == subs_id.loc[i,0])
                    try:
                        downstream_node = us_node.loc[subs_id.loc[i, 0] - 1, "US"]
                        loc = np.where(
                            distribution_properties["id"] == downstream_node
                        )[0][0]
                    except IndexError:
                        other_sub_loc = np.where(
                            replacement_sub["missing"] == subs_id.loc[i, 0]
                        )[0][0]
                        downstream_node = us_node.loc[
                            replacement_sub.loc[other_sub_loc, "replacement"] - 1, "US"
                        ]
                        loc = np.where(
                            distribution_properties["id"] == downstream_node
                        )[0][0]

                    # to get the Non-Exceedance probability for a specific Value.
                    F = gumbel_r.cdf(
                        max_values[i],
                        loc=distribution_properties.loc[loc, "loc"],
                        scale=distribution_properties.loc[loc, "scale"],
                    )
                    # then calculate the return_period (return period) return_period = 1/(1-F).
                    return_period.append(round(1 / (1 - F), 2))

            try:
                retun_period_map = np.ones((rows, cols), dtype=np.float32) * 0
                for i in range(rows):
                    for j in range(cols):
                        # print("i = " + str(i) + ", j= " + str(j))
                        if not np.isnan(max_depth[i, j]):
                            if max_depth[i, j] > 0:
                                # print("i = " + str(i) + ", j= " + str(j))
                                # if the sub basin is in the Sub ID list
                                if sub_id_map_v[i, j] in subs_id[0].tolist():
                                    # print("Sub = " + str(sub_id_map_v[i,j]))
                                    # go get the return period directly
                                    retun_period_map[i, j] = return_period[
                                        np.where(subs_id[0] == sub_id_map_v[i, j])[0][0]
                                    ]
                                else:
                                    # print("Extra  Sub = " + str(sub_id_map_v[i,j]))
                                    # the sub ID is one of the added subs not routed by RIM, so it existed in the
                                    # extra_subs list with a reference to a SubID routed by RIM.
                                    rim_sub = extra_subs.loc[
                                        np.where(
                                            extra_subs["addSub"] == sub_id_map_v[i, j]
                                        )[0][0],
                                        "rim_sub",
                                    ]
                                    retun_period_map[i, j] = return_period[
                                        np.where(subs_id[0] == rim_sub)[0][0]
                                    ]
            except:
                print("Error")
                check.append(timestep)
                klist.append(k)
                continue

            # save the return period ASCII file
            file_name = "ReturnPeriod" + str(timestep) + ".asc"

            with open(save_to + "/" + file_name, "w") as File:
                # write the first lines
                for i in range(len(spatial_ref)):
                    File.write(str(spatial_ref[i].decode()[:-2]) + "\n")

                for i in range(np.shape(retun_period_map)[0]):
                    File.writelines(list(map(self.StringSpace, retun_period_map[i, :])))
                    File.write("\n")

            # zip the file
            with zipfile.ZipFile(
                save_to + "/" + file_name[:-4] + ".zip", "w", zipfile.ZIP_DEFLATED
            ) as newzip:
                newzip.write(save_to + "/" + file_name, arcname=file_name)
            # delete the file
            os.remove(save_to + "/" + file_name)

        check = list(zip(check, klist))
        if len(check) > 0:
            np.savetxt(wpath + "CheckWaterDepth.txt", check, fmt="%6d")

    def create_trace_all(
        self,
        config_file_path,
        rim_subs_file,
        trace_file,
        us_only: int = 1,
        hydrological_inputs_file: str = "",
    ):
        """CreateTraceALL.

        Parameters
        ----------
        config_file_path: [String]
            SWIM configuration file.
        rim_subs_file: [String]
            path to text file with all the ID of SWIM sub-basins.
        trace_file: TYPE
            DESCRIPTION.
        us_only: TYPE, optional
            DESCRIPTION. The default is 1.
        hydrological_inputs_file: TYPE, optional
            DESCRIPTION. The default is ''.

        Returns
        -------
        None.
        """
        # reading the file
        config = pd.read_csv(config_file_path, header=None)
        # process the Configuration file
        # get the route rows from the file
        route = pd.DataFrame(columns=["No", "DSnode", "SubID", "USnode", "No2"])

        j = 0
        for i in range(len(config)):
            if config[0][i].split()[0] == "route":
                route.loc[j, :] = list(map(int, config[0][i].split()[1:]))
                j = j + 1

        # get RIM Sub-basins
        subs = pd.read_csv(rim_subs_file, header=None)
        subs = subs.rename(columns={0: "SubID"})

        subs["US"] = None
        subs["DS"] = None

        for i in range(len(subs)):
            try:
                # if the sub-basin is in the route array, so it is routed by rrm.
                loc = np.where(route["SubID"] == subs.loc[i, "SubID"])[0][0]
                subs.loc[i, "US"] = int(route.loc[loc, "USnode"])
                subs.loc[i, "DS"] = int(route.loc[loc, "DSnode"])
            except IndexError:
                # if the sub-basin is not in the route array, so it is not routed by rrm.
                # but still can be routed using RIM

                # subs.loc[i,'US'] = None
                # subs.loc[i,'DS'] = None
                subs.loc[i, "US"] = -1
                subs.loc[i, "DS"] = -1

        # Save the file with the same format required for the hg R code

        subs.to_csv(trace_file, index=None, header=True)
        #    ToSave = subs.loc[:,['SubID','US']]
        #    ToSave['Extra column 1'] = -1
        #    ToSave['Extra column 2'] = -1
        #    ToSave.to_csv(save_path + trace_file,header = None, index = None)
