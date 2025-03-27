import xml.etree.ElementTree as ET
import baseline_comparison as bc
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools
import plotly.graph_objects as go
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm

if __name__ == "__main__":

    result_directory = "results"

    # Paths to the example XML files
    primary = "input_data/ICEYE_X8_SLC_SLEA_1000451_20220823T200426.xml"
    secondary = "input_data/ICEYE_X2_SLC_SLEA_1000479_20220812T094148.xml"

    if not os.path.exists(primary) or not os.path.exists(secondary):
        print(
            f"File not found: {primary if not os.path.exists(primary) else secondary}"
        )
    else:
        # Extract state vectors for primary and secondary satellites
        positions_primary, _, _ = bc.extract_state_vectors(primary)
        positions_secondary, _, _ = bc.extract_state_vectors(secondary)

        # Extract X, Y, Z coordinates for primary satellite
        x_coords_primary = [pos[0] for pos in positions_primary]
        y_coords_primary = [pos[1] for pos in positions_primary]
        z_coords_primary = [pos[2] for pos in positions_primary]

        # Extract X, Y, Z coordinates for secondary satellite
        x_coords_secondary = [pos[0] for pos in positions_secondary]
        y_coords_secondary = [pos[1] for pos in positions_secondary]
        z_coords_secondary = [pos[2] for pos in positions_secondary]

        # --- Create a static 3D plot using Matplotlib ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot the primary satellite's orbit path
        ax.plot(
            x_coords_primary,
            y_coords_primary,
            z_coords_primary,
            label="Primary Satellite Orbit",
            color="blue",
        )

        # Plot the secondary satellite's orbit path
        ax.plot(
            x_coords_secondary,
            y_coords_secondary,
            z_coords_secondary,
            label="Secondary Satellite Orbit",
            color="red",
        )

        # Add labels and title
        ax.set_xlabel("X Coordinate (m)")
        ax.set_ylabel("Y Coordinate (m)")
        ax.set_zlabel("Z Coordinate (m)")
        ax.set_title("3D Orbit Paths of Satellites")

        # Add a legend
        ax.legend()

        # Save the Matplotlib plot as a PNG file
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        png_output_file = os.path.join(result_directory, "satellite_orbits.png")
        plt.savefig(png_output_file)
        print(f"Static 3D plot saved to {png_output_file}")

        # Show the plot
        plt.show()

        # --- Create an interactive 3D plot using Plotly ---
        fig_plotly = go.Figure()

        # Add the primary satellite's orbit path
        fig_plotly.add_trace(
            go.Scatter3d(
                x=x_coords_primary,
                y=y_coords_primary,
                z=z_coords_primary,
                mode="lines",
                line=dict(color="blue", width=2),
                name="Primary Satellite Orbit",
            )
        )

        # Add the secondary satellite's orbit path
        fig_plotly.add_trace(
            go.Scatter3d(
                x=x_coords_secondary,
                y=y_coords_secondary,
                z=z_coords_secondary,
                mode="lines",
                line=dict(color="red", width=2),
                name="Secondary Satellite Orbit",
            )
        )

        # Add labels and title
        fig_plotly.update_layout(
            scene=dict(
                xaxis_title="X Coordinate (m)",
                yaxis_title="Y Coordinate (m)",
                zaxis_title="Z Coordinate (m)",
            ),
            title="3D Orbit Paths of Satellites",
        )

        # Save the Plotly plot as an HTML file
        html_output_file = os.path.join(result_directory, "interactive_orbits.html")
        fig_plotly.write_html(html_output_file)
        print(f"Interactive 3D plot saved to {html_output_file}")

        # Show the interactive Plotly plot
        fig_plotly.show()
