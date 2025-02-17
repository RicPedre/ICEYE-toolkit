import baseline_comparison as bc
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

if __name__ == "__main__":
    # Directory definitions
    input_directory = "input_data"
    result_directory = "results"

    # Get the XML files to compare
    xml_files = bc.get_xml_files(input_directory)

    # Initialize the graph
    G = nx.Graph()

    # Reference acquisition (first in the list)
    reference = xml_files[0]
    P_ref, u_f_ref, time_ref = bc.extract_state_vector(reference)
    time_ref_dt = datetime.strptime(time_ref, "%Y-%m-%dT%H:%M:%S.%f")

    # Add the reference node
    G.add_node(reference, pos=(0, 0), label="Reference")

    # Compare each pair of satellite acquisitions
    for i in tqdm(range(1, len(xml_files)), desc="Comparing acquisitions"):
        secondary = xml_files[i]

        # Extract state vector for the secondary image
        P_secondary, _, time_secondary = bc.extract_state_vector(secondary)
        time_secondary_dt = datetime.strptime(time_secondary, "%Y-%m-%dT%H:%M:%S.%f")

        # Build the Line of Sight (LoS) vector for the reference image
        incidence, azimuth = bc.extract_angles(reference)
        u_LOS = bc.compute_u_LOS(incidence, azimuth)

        # Compute the perpendicular baseline
        _, _, _, B_perp_sign = bc.compute_perpendicular_baseline(
            P_ref, P_secondary, u_LOS, u_f_ref
        )

        # Compute the temporal baseline
        temporal_baseline = (time_secondary_dt - time_ref_dt).total_seconds() / 86400

        # Add the secondary node
        G.add_node(
            secondary, pos=(temporal_baseline, B_perp_sign), label=f"Acquisition {i+1}"
        )

        # Add an edge between the reference and the secondary node
        G.add_edge(reference, secondary)

    # Get positions for all nodes
    pos = nx.get_node_attributes(G, "pos")

    # Draw the network
    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=500,
        node_color="skyblue",
        font_size=10,
        font_weight="bold",
        edge_color="gray",
    )
    labels = nx.get_node_attributes(G, "label")
    nx.draw_networkx_labels(G, pos, labels, font_size=12)

    plt.xlabel("Temporal Baseline (days)", fontsize=14)
    plt.ylabel("Perpendicular Baseline", fontsize=14)
    plt.title("Network Plot of Satellite Acquisitions", fontsize=16)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to the results directory
    plt.savefig(f"{result_directory}/network_plot.png")
    plt.close()
