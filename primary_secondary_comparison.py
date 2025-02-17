import baseline_comparison as bc
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

if __name__ == "__main__":

    # Directory definitions
    input_directory = "input_data"
    result_directory = "results"

    # Get the XML files to compare
    xml_files = bc.get_xml_files(input_directory)

    # Compare each pair of satellite acquisitions
    for i in tqdm(range(len(xml_files)), desc="Comparing satellite acquisition"):
        primary = xml_files[i]

        # Extract state vector for the primary image
        P_primary, u_f_primary, time_primary = bc.extract_state_vector(primary)
        time_primary_dt = datetime.strptime(time_primary, "%Y-%m-%dT%H:%M:%S.%f")

        # Initialize lists to store times and perpendicular baselines
        times = [time_primary_dt, None]
        baselines_perpendicular = [0, None]

        # Create a new figure for the plot
        plt.figure(figsize=(12, 8))

        for j in range(len(xml_files)):
            secondary = xml_files[j]

            # Extract state vector for the secondary image
            P_secondary, _, time_secondary = bc.extract_state_vector(secondary)
            time_secondary_dt = datetime.strptime(
                time_secondary, "%Y-%m-%dT%H:%M:%S.%f"
            )

            # Build the Line of Sight (LoS) vector for the primary image
            incidence, azimuth = bc.extract_angles(primary)
            u_LOS = bc.compute_u_LOS(incidence, azimuth)

            # Compute the perpendicular baseline
            _, _, _, B_perp_sign = bc.compute_perpendicular_baseline(
                P_primary, P_secondary, u_LOS, u_f_primary
            )

            # Store the time of acquisition and perpendicular baseline
            times[1] = time_secondary_dt
            baselines_perpendicular[1] = B_perp_sign

            # Plot the data points for the secondary image
            plt.plot(
                times,
                baselines_perpendicular,
                marker="o",
                mfc="blue",
                mec="blue",
                color="blue",
                linestyle="-",
                label="Secondary Image" if j == 0 else "",
            )

        # Highlight the primary image
        plt.scatter(
            time_primary_dt, 0, color="red", s=100, zorder=5, label="Primary Image"
        )
        plt.axvline(x=time_primary_dt, color="red", linestyle="--")
        plt.axhline(y=0, color="red", linestyle="--")

        # Set labels, title, and grid for the plot
        plt.xlabel("Time of Acquisition", fontsize=14)
        plt.ylabel("Perpendicular Baseline", fontsize=14)
        plt.title(f"Scatter Plot for Primary Image {i+1}", fontsize=16)
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        # Save the plot to the results directory
        plt.savefig(f"{result_directory}/scatter_plot_primary_{i+1}.png")
        plt.close()
