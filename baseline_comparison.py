import xml.etree.ElementTree as ET
import numpy as np
import math
import os
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm


def extract_state_vectors(xml_file: str):
    """
    Extract all state vectors (satellite positions) from an ICEYE metadata XML file.

    Parameters:
      xml_file (str): Path to the metadata XML file.

    Returns:
      positions (list of np.ndarray): A list of numpy arrays containing the 3D positions [posX, posY, posZ].
      flight_directions (list of np.ndarray): A list of unit vectors representing the flight directions.
      time_strings (list of str): A list of timestamp strings for the state vectors.
    """
    # Check if the file exists
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"File not found: {xml_file}")

    try:
        # Parse the XML file
        tree = ET.parse(xml_file)
    except ET.ParseError as e:
        # Raise an error if the XML file cannot be parsed
        raise ValueError(f"Error parsing XML file: {e}")

    # Get the root element of the XML tree
    root = tree.getroot()

    # Find all state vector elements in the XML file
    state_vectors = root.findall(".//Orbit_State_Vectors/orbit_vector")
    if not state_vectors:
        # Raise an error if no state vectors are found
        raise ValueError("No state vectors found in the metadata file.")

    # Initialize lists to store positions, flight directions, and timestamps
    positions = []
    flight_directions = []
    time_strings = []

    # Iterate over each state vector element
    for sv in state_vectors:
        try:
            # Extract position components (posX, posY, posZ)
            posX = float(sv.find("posX").text)
            posY = float(sv.find("posY").text)
            posZ = float(sv.find("posZ").text)

            # Extract the timestamp
            time_str = sv.find("time").text

            # Extract velocity components (velX, velY, velZ)
            velX = float(sv.find("velX").text)
            velY = float(sv.find("velY").text)
            velZ = float(sv.find("velZ").text)

            # Compute the velocity vector
            velocity_vector = np.array([velX, velY, velZ])

            # Normalize the velocity vector to get the flight direction
            flight_direction = velocity_vector / np.linalg.norm(velocity_vector)

            # Append the position, flight direction, and timestamp to their respective lists
            positions.append(np.array([posX, posY, posZ]))
            flight_directions.append(flight_direction)
            time_strings.append(time_str)
        except AttributeError as e:
            # Raise an error if any expected XML element is missing
            raise ValueError(f"Missing expected XML elements: {e}")

    # Return the extracted positions, flight directions, and timestamps
    return positions, flight_directions, time_strings


def extract_angles(xml_file: str):
    """
    Extract look and azimuth angles from an ICEYE metadata XML file.
    This example assumes that the metadata contains the look and azimuth angles under the tag 'Look'.

    Parameters:
      xml_file (str): Path to the metadata XML file.

    Returns:
      look_angle (float): The look angle in degrees.
      azimuth_angle (float): The azimuth angle in degrees.

    """
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f"File not found: {xml_file}")

    try:
        tree = ET.parse(xml_file)
    except ET.ParseError as e:
        raise ValueError(f"Error parsing XML file: {e}")

    root = tree.getroot()
    # Extract look center and satellite look angle from the XML elements.
    try:
        look_angle = float(root.find(".//satellite_look_angle").text)
        azimuth_angle = float(root.find(".//heading").text)
    except AttributeError as e:
        raise ValueError(f"Missing expected XML elements: {e}")

    return look_angle, azimuth_angle


def compute_u_LOS(look_angle_deg: float, azimuth_angle_deg: float):
    """
    Computes the LOS unit vector using look and azimuth angles.
    Formula:
       u_LOS = [ sin(theta)*cos(phi),
                 sin(theta)*sin(phi),
                 cos(theta) ]
    where theta is the look angle (in radians) and phi is the azimuth angle (in radians).

    Parameters:
        look_angle_deg (float): The look angle in degrees.
        azimuth_angle_deg (float): The azimuth angle in degrees.

    Returns:
        u (np.ndarray): The unit vector representing the line-of-sight direction.

    """
    theta = math.radians(look_angle_deg)
    phi = math.radians(azimuth_angle_deg)
    u = np.array(
        [
            math.cos(theta) * math.sin(phi),
            math.cos(theta) * math.cos(phi),
            math.sin(theta),
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
        B_perp_signed_magnitude (float): The signed magnitude of the perpendicular vector.

    """

    # Compute the full baseline vector.
    B = P2 - P1
    B_total = np.linalg.norm(B)

    # Compute the perpendicular component of the baseline.
    B_parallel = np.dot(B, u_LOS)
    B_perp = B - B_parallel * u_LOS

    # Another way to calculate the perpendicular component
    theta = math.acos(np.dot(B, u_LOS) / B_total)
    B_perp = B * math.sin(theta)
    sign = np.sign(np.dot(B_perp, u_flight))
    B_perp_magnitude = np.linalg.norm(B_perp)
    B_perp_signed_magnitude = B_perp_magnitude * sign

    return B_total, B_perp, B_perp_magnitude, B_perp_signed_magnitude


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
    target_baseline = 200000  # Target perpendicular baseline magnitude in meters.

    comparison_results = []

    plt.figure(figsize=(12, 8))

    for i in tqdm(range(len(xml_files)), desc="Processing files"):
        for j in range(i + 1, len(xml_files)):
            primary_metadata_file = xml_files[i]
            secondary_metadata_file = xml_files[j]

            # Extract all state vectors for primary and secondary.
            P_primary_list, u_flight_primary_list, time_primary_list = (
                extract_state_vectors(primary_metadata_file)
            )
            P_secondary_list, u_flight_secondary_list, time_secondary_list = (
                extract_state_vectors(secondary_metadata_file)
            )

            # Compare state vectors by matching indices (first with first, second with second, etc.).
            min_perpendicular_baseline = float("inf")
            max_perpendicular_baseline = float("-inf")
            min_result = None
            max_result = None

            # Extract look and azimuth angles for the primary state vector.
            look_primary, azimuth_primary = extract_angles(primary_metadata_file)

            # Extract look and azimuth angles for the secondary state vector.
            look_secondary, azimuth_secondary = extract_angles(secondary_metadata_file)

            diff_look = abs(look_primary - look_secondary)
            diff_azimuth = abs(azimuth_primary - azimuth_secondary)

            # Calculate the LoS vector for the primary state vector.
            u_LOS_primary = compute_u_LOS(look_primary, azimuth_primary)

            # Iterate over each state vector in the metadata file
            for k, (P_primary, u_flight_primary, time_primary) in enumerate(
                zip(P_primary_list, u_flight_primary_list, time_primary_list)
            ):
                if k >= len(P_secondary_list):
                    # Skip if the secondary list has fewer state vectors.
                    break

                # Get the corresponding state vector from the secondary metadata.
                P_secondary = P_secondary_list[k]
                time_secondary = time_secondary_list[k]

                # Compute the baseline and its perpendicular component.
                (
                    B_total,
                    B_perp_vector,
                    B_perp_magnitude,
                    B_perp_signed_magnitude,
                ) = compute_perpendicular_baseline(
                    P_primary, P_secondary, u_LOS_primary, u_flight_primary
                )

                # Convert time strings to datetime objects.
                time_primary_dt = datetime.strptime(
                    time_primary, "%Y-%m-%dT%H:%M:%S.%f"
                )
                time_secondary_dt = datetime.strptime(
                    time_secondary, "%Y-%m-%dT%H:%M:%S.%f"
                )

                # Compute temporal baseline in days.
                temporal_baseline = abs(
                    (time_primary_dt - time_secondary_dt).total_seconds() / 86400
                )

                if B_perp_magnitude < min_perpendicular_baseline:
                    min_perpendicular_baseline = B_perp_magnitude
                    min_state_vector_index = k

                if B_perp_magnitude > max_perpendicular_baseline:
                    max_perpendicular_baseline = B_perp_magnitude
                    max_state_vector_index = k

            comparison_results.append(
                {
                    "primary_file": primary_metadata_file,
                    "secondary_file": secondary_metadata_file,
                    "minimum_state_vector_index": min_state_vector_index,
                    "minimum_perpendicular_baseline_magnitude": min_perpendicular_baseline,
                    "maximum_state_vector_index": max_state_vector_index,
                    "maximum_perpendicular_baseline_magnitude": max_perpendicular_baseline,
                    "temporal_baseline_days": temporal_baseline,
                    "diff_look": diff_look,
                    "diff_azimuth": diff_azimuth,
                }
            )

    # Save results to a CSV file.
    csv_file = os.path.join(result_directory, "comparison_results.csv")
    with open(csv_file, mode="w", newline="") as file:
        # Update the fieldnames to include all keys in comparison_results
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "primary_file",
                "secondary_file",
                "temporal_baseline_days",
                "minimum_state_vector_index",
                "minimum_perpendicular_baseline_magnitude",
                "maximum_state_vector_index",
                "maximum_perpendicular_baseline_magnitude",
                "diff_look",
                "diff_azimuth",
            ],
        )
        writer.writeheader()
        for result in comparison_results:
            writer.writerow(result)

    # # Create a graph.
    # temporal_baselines = [
    #     result["temporal_baseline_days"] for result in comparison_results
    # ]
    # perpendicular_baselines = [
    #     result["perpendicular_baseline_magnitude"] for result in comparison_results
    # ]

    # colors = [
    #     "green" if baseline < target_baseline else "red"
    #     for baseline in perpendicular_baselines
    # ]
    # plt.scatter(temporal_baselines, perpendicular_baselines, c=colors)
    # plt.xlabel("Temporal Baseline (days)")
    # plt.ylabel("Perpendicular Baseline Magnitude (m)")
    # plt.title("Temporal Baseline vs Perpendicular Baseline Magnitude")
    # plt.grid(True)
    # plt.savefig(os.path.join(result_directory, "baseline_graph.png"))
    # plt.close()
