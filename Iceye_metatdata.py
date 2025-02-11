import xml.etree.ElementTree as ET
import numpy as np
import math
import os
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm


def extract_state_vector(xml_file, ref="center"):
    """
    Extract a state vector (satellite position) from an ICEYE metadata XML file.
    This example assumes that the metadata contains a set of state vectors under the tag 'Orbit_State_Vectors'
    with sub-elements for the time (state_vector_time_utc) and position components (posX, posY, posZ).

    Parameters:
      xml_file: Path to the metadata XML file.
      ref: Reference selection; here we choose the middle state vector (i.e. scene center).

    Returns:
      position: A numpy array containing the 3D position [posX, posY, posZ].
      time_str: The timestamp string for the selected state vector.

    (See ICEYE Product Metadata documentation for the list of metadata elements :contentReference[oaicite:2]{index=2}.)
    """
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"File not found: {xml_file}")

    try:
        tree = ET.parse(xml_file)
    except ET.ParseError as e:
        raise ValueError(f"Error parsing XML file: {e}")

    root = tree.getroot()
    # Find all state vector elements. Adjust the XPath as needed to match your metadata file structure.
    state_vectors = root.findall(".//Orbit_State_Vectors/orbit_vector")
    if not state_vectors:
        raise ValueError("No state vectors found in the metadata file.")

    # Choose the reference state vector; here we use the middle one.
    idx = len(state_vectors) // 2
    sv = state_vectors[idx]

    # Extract position components from the XML elements.
    try:
        posX = float(sv.find("posX").text)
        posY = float(sv.find("posY").text)
        posZ = float(sv.find("posZ").text)
        time_str = sv.find("time").text
    except AttributeError as e:
        raise ValueError(f"Missing expected XML elements: {e}")

    return np.array([posX, posY, posZ]), time_str


def extract_angles(xml_file):
    """
    Extract incidence and azimuth angles from an ICEYE metadata XML file.
    This example assumes that the metadata contains the incidence and azimuth angles under the tag 'Look'.

    Parameters:
      xml_file: Path to the metadata XML file.

    Returns:
      incidence_angle: The incidence angle in degrees.
      azimuth_angle: The azimuth angle in degrees.

    (See ICEYE Product Metadata documentation for the list of metadata elements :contentReference[oaicite:2]{index=2}.)
    """
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"File not found: {xml_file}")

    try:
        tree = ET.parse(xml_file)
    except ET.ParseError as e:
        raise ValueError(f"Error parsing XML file: {e}")

    root = tree.getroot()
    # Extract incidence center and satellite look angle from the XML elements.
    try:
        incidence_angle = float(root.find(".//incidence_center").text)
        azimuth_angle = float(root.find(".//satellite_look_angle").text)
    except AttributeError as e:
        raise ValueError(f"Missing expected XML elements: {e}")

    return incidence_angle, azimuth_angle


def compute_u_LOS(incidence_angle_deg, azimuth_angle_deg):
    """
    Computes the LOS unit vector using incidence and azimuth angles.
    Formula:
       u_LOS = [ sin(theta)*cos(phi),
                 sin(theta)*sin(phi),
                 cos(theta) ]
    where theta is the incidence angle (in radians) and phi is the azimuth angle (in radians).
    """
    theta = math.radians(incidence_angle_deg)
    phi = math.radians(azimuth_angle_deg)
    u = np.array(
        [
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ]
    )
    return u / np.linalg.norm(u)


def average_LOS(u1, u2):
    """
    Compute the average LOS vector from two LOS vectors.
    This simple average is then normalized.
    """
    u_avg = u1 + u2
    return u_avg / np.linalg.norm(u_avg)


def compute_perpendicular_baseline(P1, P2, u_LOS):
    """
    Computes the full baseline vector and then its perpendicular component.
    B = P2 - P1
    B_parallel = (B ⋅ u_LOS) * u_LOS
    B_perp = B - B_parallel
    Returns the full baseline magnitude, the perpendicular vector, and its magnitude.
    """
    B = P2 - P1
    B_total = np.linalg.norm(B)
    B_parallel = np.dot(B, u_LOS) * u_LOS
    B_perp = B - B_parallel
    B_perp_norm = np.linalg.norm(B_perp)
    return B_total, B_perp, B_perp_norm


def get_xml_files(directory):
    """
    Get a list of all XML files in the given directory.
    """
    return [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".xml")
    ]


if __name__ == "__main__":
    # Directory containing the metadata XML files.
    input_directory = "input_data"
    result_directory = "results"

    # Get all XML files in the directory.
    xml_files = get_xml_files(input_directory)

    # List to store comparison results.
    comparison_results = []

    # Compare each pair of XML files.
    for i in tqdm(range(len(xml_files)), desc="Processing files"):
        for j in range(i + 1, len(xml_files)):
            primary_metadata_file = xml_files[i]
            secondary_metadata_file = xml_files[j]

            # Extract state vectors for primary and secondary.
            P_primary, time_primary = extract_state_vector(primary_metadata_file)
            P_secondary, time_secondary = extract_state_vector(secondary_metadata_file)

            # Extract incidence and azimuth angles for primary and secondary.
            incidence_primary, azimuth_primary = extract_angles(primary_metadata_file)
            incidence_secondary, azimuth_secondary = extract_angles(
                secondary_metadata_file
            )

            # Calculation of the LoS vectors
            u_LOS_primary = compute_u_LOS(incidence_primary, azimuth_primary)
            u_LOS_secondary = compute_u_LOS(incidence_secondary, azimuth_secondary)

            # Compute the average LOS vector to use as the common reference.
            u_LOS_avg = average_LOS(u_LOS_primary, u_LOS_secondary)

            # Compute the baseline and its perpendicular component using the average LOS.
            B_total, B_perp_vector, B_perp_magnitude = compute_perpendicular_baseline(
                P_primary, P_secondary, u_LOS_avg
            )

            # Convert time strings to datetime objects
            time_primary_dt = datetime.strptime(time_primary, "%Y-%m-%dT%H:%M:%S.%f")
            time_secondary_dt = datetime.strptime(
                time_secondary, "%Y-%m-%dT%H:%M:%S.%f"
            )

            # Compute temporal baseline
            temporal_baseline = abs(
                (time_primary_dt - time_secondary_dt).total_seconds() / 86400
            )

            # Store the comparison result.
            comparison_results.append(
                {
                    "primary_file": primary_metadata_file,
                    "secondary_file": secondary_metadata_file,
                    "perpendicular_baseline_magnitude": B_perp_magnitude,
                    "temporal_baseline_days": temporal_baseline,
                }
            )

    # Save results to a CSV file.
    csv_file = os.path.join(result_directory, "comparison_results.csv")
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "primary_file",
                "secondary_file",
                "perpendicular_baseline_magnitude",
                "temporal_baseline_days",
            ],
        )
        writer.writeheader()
        for result in comparison_results:
            writer.writerow(result)

    # # Print all comparison results.
    # for result in comparison_results:
    #     print(f"Comparing {result['primary_file']} and {result['secondary_file']}")
    #     print(
    #         "Perpendicular baseline magnitude (m):",
    #         round(result["perpendicular_baseline_magnitude"], 2),
    #     )
    #     print("Temporal baseline (days):", round(result["temporal_baseline_days"], 2))
    #     print("\n")

    # Create a graph.
    temporal_baselines = [
        result["temporal_baseline_days"] for result in comparison_results
    ]
    perpendicular_baselines = [
        result["perpendicular_baseline_magnitude"] for result in comparison_results
    ]

    plt.scatter(temporal_baselines, perpendicular_baselines)
    plt.xlabel("Temporal Baseline (days)")
    plt.ylabel("Perpendicular Baseline Magnitude (m)")
    plt.title("Temporal Baseline vs Perpendicular Baseline Magnitude")
    plt.grid(True)
    plt.savefig(os.path.join(result_directory, "baseline_graph.png"))
