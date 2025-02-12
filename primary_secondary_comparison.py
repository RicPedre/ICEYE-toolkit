import baseline_comparison as bc
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

if __name__ == "__main__":

    # directory definitions
    input_directory = "input_data"
    result_directory = "results"

    # get the xml file to compare
    xml_files = bc.get_xml_files(input_directory)

    # compare each pair of satellite acquisition
    for i in tqdm(range(len(xml_files)), desc="Comparing satellite acquisition"):
        primary = xml_files[i]
        P_primary, u_f_primary, time_primary = bc.extract_state_vector(primary)
        time_primary_dt = datetime.strptime(time_primary, "%Y-%m-%dT%H:%M:%S.%f")

        times = [time_primary_dt, None]
        baselines_perpendicular = [0, None]

        plt.figure(figsize=(10, 6))

        for j in range(len(xml_files)):
            secondary = xml_files[j]

            # Extract state vectors
            P_secondary, _, time_secondary = bc.extract_state_vector(secondary)
            time_secondary_dt = datetime.strptime(
                time_secondary, "%Y-%m-%dT%H:%M:%S.%f"
            )

            # Build the LoS vector for the primary image
            incidence, azimuth = bc.extract_angles(primary)
            u_LOS = bc.compute_u_LOS(incidence, azimuth)

            # Compute the perpendicular baseline
            _, _, _, B_perp_sign = bc.compute_perpendicular_baseline(
                P_primary, P_secondary, u_LOS, u_f_primary
            )

            # Store the time of acquisition and perpendicular baseline
            times[1] = time_secondary_dt
            baselines_perpendicular[1] = B_perp_sign

            plt.plot(times, baselines_perpendicular, label=f"Secondary Image {j+1}")

        plt.savefig(f"{result_directory}/scatter_plot_primary_{i+1}.png")
        plt.close()

        # # Plot the scatter plot
        # plt.scatter(times_secondary, baselines_perpendicular, label="Secondary Images")
        # plt.scatter(time_primary_dt, 0, color="r", label="Primary Image")
        # plt.xlabel("Time of Acquisition (Secondary)")
        # plt.ylabel("Perpendicular Baseline")
        # plt.title(f"Scatter Plot for Primary Image {i+1}")
        # plt.legend()
        # plt.grid(True)

        # # Set the limits to center the primary data in the plot for better visualization
        # if times_secondary:
        #     plt.xlim(
        #         min(times_secondary + [time_primary_dt]),
        #         max(times_secondary + [time_primary_dt]),
        #     )
        # else:
        #     plt.xlim(time_primary_dt, time_primary_dt)

        # if valid_baselines:
        #     plt.ylim(min(valid_baselines), max(valid_baselines))
        # else:
        #     plt.ylim(0, 0)

        # plt.savefig(f"{result_directory}/scatter_plot_primary_{i+1}.png")
        # plt.close()
