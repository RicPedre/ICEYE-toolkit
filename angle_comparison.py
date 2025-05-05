import baseline_comparison as bs
import matplotlib.pyplot as plt
import os
import csv

imput_directory = "input_data"
result_directory = "results"
look_angle = []
azimuth_angle = []
csv_data = []

# Get metadata files
metadata_files = bs.get_xml_files(imput_directory)

# Extract look and azimuth angles
for i in range(len(metadata_files)):
    look, azimuth = bs.extract_angles(metadata_files[i])
    look_direction = bs.extract_look_direction(metadata_files[i])
    orbit_direction = bs.extract_orbit_direction(metadata_files[i])

    csv_data.append(
        {
            "filename": metadata_files[i],
            "look_angle": look,
            "azimuth_angle": azimuth,
            "look_direction": look_direction,
            "orbit_direction": orbit_direction,
        }
    )
    look_angle.append(look)
    azimuth_angle.append(azimuth)

# Write to CSV
csv_file = os.path.join(result_directory, "angle_data.csv")
with open(csv_file, mode="w", newline="") as file:
    fieldnames = [
        "filename",
        "look_angle",
        "azimuth_angle",
        "look_direction",
        "orbit_direction",
    ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for row in csv_data:
        writer.writerow(row)


# Create a histogram for look angles
plt.figure(figsize=(10, 6))
plt.hist(
    look_angle,
    bins=range(int(min(look_angle)), int(max(look_angle)) + 1),
    color="blue",
    alpha=0.7,
    edgecolor="black",
)
plt.xlabel("Look Angle (degrees)")
plt.ylabel("Frequency")
plt.title("Histogram of Look Angles from Metadata Files")

# Save the histogram to the results directory
output_file = os.path.join(result_directory, "look_angles_histogram.png")
plt.savefig(output_file)
print(f"Histogram saved to {output_file}")

# Show the histogram
plt.show()
