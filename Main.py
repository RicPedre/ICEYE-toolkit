import xml.etree.ElementTree as ET
import numpy as np
import math
import os
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm


def extract_state_vector(xml_file: str, ref: str = "center"):
    """
    Extract a state vector (satellite position) from an ICEYE metadata XML file.
    This example assumes that the metadata contains a set of state vectors under the tag 'Orbit_State_Vectors'
    with sub-elements for the time (state_vector_time_utc), position components (posX, posY, posZ)
    and velocity components (posX, posY, posZ).

    Parameters:
      xml_file (str): Path to the metadata XML file.
      ref (str): Reference selection; here we choose the middle state vector (i.e. scene center).

    Returns:
      position (np.ndarray): A numpy array containing the 3D position [posX, posY, posZ].
      flight_direction (np.ndarray): The unit vector representing the flight direction.
      time_str (str): The timestamp string for the selected state vector.

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

    # extract the velocity components
    try:
        velX = float(sv.find("velX").text)
        velY = float(sv.find("velY").text)
        velZ = float(sv.find("velZ").text)
    except AttributeError as e:
        raise ValueError(f"Missing expected XML elements: {e}")

    # compute the flight direction unit vector:
    velocity_vector = np.array([velX, velY, velZ])
    flight_direction = velocity_vector / np.linalg.norm(velocity_vector)

    return np.array([posX, posY, posZ]), flight_direction, time_str


def extract_angles(xml_file: str):
    """
    Extract incidence and azimuth angles from an ICEYE metadata XML file.
    This example assumes that the metadata contains the incidence and azimuth angles under the tag 'Look'.

    Parameters:
      xml_file (str): Path to the metadata XML file.

    Returns:
      incidence_angle (float): The incidence angle in degrees.
      azimuth_angle (float): The azimuth angle in degrees.

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
        azimuth_angle = float(root.find(".//heading").text)
    except AttributeError as e:
        raise ValueError(f"Missing expected XML elements: {e}")

    return incidence_angle, azimuth_angle


def compute_u_LOS(incidence_angle_deg: float, azimuth_angle_deg: float):
    """
    Computes the LOS unit vector using incidence and azimuth angles.
    Formula:
       u_LOS = [ sin(theta)*cos(phi),
                 sin(theta)*sin(phi),
                 cos(theta) ]
    where theta is the incidence angle (in radians) and phi is the azimuth angle (in radians).

    Parameters:
        incidence_angle_deg (float): The incidence angle in degrees.
        azimuth_angle_deg (float): The azimuth angle in degrees.

    Returns:
        u (np.ndarray): The unit vector representing the line-of-sight direction.

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


def compute_perpendicular_baseline(
    P1: np.ndarray, P2: np.ndarray, u_LOS: np.ndarray, u_flight: np.ndarray
):
    """
    Computes the full baseline vector and then its perpendicular component, following the formulas:
    B = P2 - P1
    B_parallel = (B ⋅ u_LOS)
    B_perp = square_root( ||B||^2 - B_parallel^2 )

    Parameters:
        P1 (np.ndarray): The position vector of the first satellite.
        P2 (np.ndarray): The position vector of the second satellite.
        u_LOS (np.ndarray): The line-of-sight unit vector of the primary satellite.

    Returns:
        B_total (float): The full baseline magnitude.
        B_perp (np.ndarray): The perpendicular vector.
        B_perp_norm (float): The magnitude of the perpendicular vector.

    """

    # Compute the full baseline vector.
    B = P2 - P1
    B_total = np.linalg.norm(B)

    # Compute the perpendicular component of the baseline.
    # B_parallel = np.dot(B, u_LOS)
    # B_perp = np.sqrt((B_total**2) - B_parallel**2)
    # B_perp_norm = np.linalg.norm(B_perp) * sign

    # Another way to calculate the perpendicular component
    theta = math.acos(np.dot(B, u_LOS) / B_total)
    B_perp = B * math.sin(theta)
    sign = np.sign(np.dot(B_perp, u_flight))
    B_perp_magnitude = np.linalg.norm(B_perp)
    B_perp_sign = B_perp_magnitude * sign

    return B_total, B_perp, B_perp_magnitude, B_perp_sign


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
            P_primary, u_flight_primary, time_primary = extract_state_vector(
                primary_metadata_file
            )
            P_secondary, u_flight_secondary, time_secondary = extract_state_vector(
                secondary_metadata_file
            )

            # Extract incidence and azimuth angles for primary and secondary.
            incidence_primary, azimuth_primary = extract_angles(primary_metadata_file)
            incidence_secondary, azimuth_secondary = extract_angles(
                secondary_metadata_file
            )

            # Calculation of the LoS vectors
            u_LOS_primary = compute_u_LOS(incidence_primary, azimuth_primary)
            u_LOS_secondary = compute_u_LOS(incidence_secondary, azimuth_secondary)

            # Compute the baseline and its perpendicular component using the average LOS. chnaged to the primary to see what happens
            B_total, B_perp_vector, B_perp_magnitude, B_perp_abs = (
                compute_perpendicular_baseline(
                    P_primary, P_secondary, u_LOS_primary, u_flight_primary
                )
            )

            # Convert time strings to datetime objects
            time_primary_dt = datetime.strptime(time_primary, "%Y-%m-%dT%H:%M:%S.%f")
            time_secondary_dt = datetime.strptime(
                time_secondary, "%Y-%m-%dT%H:%M:%S.%f"
            )

            # Compute temporal baseline in days (assuming 1 day = 86400 seconds).
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
    plt.close()
