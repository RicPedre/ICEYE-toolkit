import baseline_comparison as bc
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

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
        positions_primary, _, time_primary = bc.extract_state_vectors(primary)
        positions_secondary, _, time_secondary = bc.extract_state_vectors(secondary)

        # Extract X, Y, Z coordinates for primary satellite
        x_coords_primary = [pos[0] for pos in positions_primary]
        y_coords_primary = [pos[1] for pos in positions_primary]
        z_coords_primary = [pos[2] for pos in positions_primary]

        # Extract X, Y, Z coordinates for secondary satellite
        x_coords_secondary = [pos[0] for pos in positions_secondary]
        y_coords_secondary = [pos[1] for pos in positions_secondary]
        z_coords_secondary = [pos[2] for pos in positions_secondary]

        # Extract relative time steps for primary and secondary satellites
        # Convert the timestamps to seconds from the start of the acquisition
        time_primary = [
            (
                datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f")
                - datetime.strptime(time_primary[0], "%Y-%m-%dT%H:%M:%S.%f")
            ).total_seconds()
            for time in time_primary
        ]
        time_secondary = [
            (
                datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f")
                - datetime.strptime(time_secondary[0], "%Y-%m-%dT%H:%M:%S.%f")
            ).total_seconds()
            for time in time_secondary
        ]

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

        # Add the Earth as a plane at sea level
        earth_radius = 0
        plane_size = (
            max(
                max(x_coords_primary + x_coords_secondary)
                - min(x_coords_primary + x_coords_secondary),
                max(y_coords_primary + y_coords_secondary)
                - min(y_coords_primary + y_coords_secondary),
            )
            * 1.5
        )  # Scale the plane size to cover the orbit paths

        x_plane = np.linspace(-plane_size, plane_size, 100)
        y_plane = np.linspace(-plane_size, plane_size, 100)
        x_plane, y_plane = np.meshgrid(x_plane, y_plane)
        z_plane = np.full_like(x_plane, earth_radius)

        ax.plot_surface(
            x_plane,
            y_plane,
            z_plane,
            color="green",
            alpha=0.5,
            label="Earth Surface",
        )

        # Add labels and title
        ax.set_xlabel("X Coordinate (m)")
        ax.set_ylabel("Y Coordinate (m)")
        ax.set_zlabel("Z Coordinate (m)")
        ax.set_title("3D Orbit Paths of Satellites with Earth Surface")

        # --- Matplotlib: Set equal aspect ratio ---
        max_range = (
            max(
                max(x_coords_primary + x_coords_secondary)
                - min(x_coords_primary + x_coords_secondary),
                max(y_coords_primary + y_coords_secondary)
                - min(y_coords_primary + y_coords_secondary),
                max(z_coords_primary + z_coords_secondary)
                - min(z_coords_primary + z_coords_secondary),
            )
            / 2.0
        )

        mid_x = (
            max(x_coords_primary + x_coords_secondary)
            + min(x_coords_primary + x_coords_secondary)
        ) / 2.0
        mid_y = (
            max(y_coords_primary + y_coords_secondary)
            + min(y_coords_primary + y_coords_secondary)
        ) / 2.0
        mid_z = (
            max(z_coords_primary + z_coords_secondary)
            + min(z_coords_primary + z_coords_secondary)
        ) / 2.0

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Add a legend
        ax.legend()

        # Save the Matplotlib plot as a PNG file
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
        png_output_file = os.path.join(
            result_directory, "satellite_orbits_with_earth.png"
        )
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
                line=dict(color=time_primary, colorscale="Viridis", width=2),
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
                line=dict(color=time_secondary, colorscale="Magma", width=2),
                name="Secondary Satellite Orbit",
            )
        )

        # Add the Earth as a plane
        fig_plotly.add_trace(
            go.Surface(
                x=x_plane,
                y=y_plane,
                z=z_plane,
                colorscale="Greens",
                opacity=0.5,
                name="Earth Surface",
            )
        )

        # Add labels and title
        fig_plotly.update_layout(
            scene=dict(
                xaxis_title="X Coordinate (m)",
                yaxis_title="Y Coordinate (m)",
                zaxis_title="Z Coordinate (m)",
                aspectmode="cube",  # Ensures equal scaling for all axes
            ),
            title="3D Orbit Paths of Satellites with Earth Surface",
        )

        # Save the Plotly plot as an HTML file
        html_output_file = os.path.join(
            result_directory, "interactive_orbits_with_earth.html"
        )
        fig_plotly.write_html(html_output_file)
        print(f"Interactive 3D plot saved to {html_output_file}")

        # Show the interactive Plotly plot
        fig_plotly.show()
