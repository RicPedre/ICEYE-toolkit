import baseline_comparison as bs
import matplotlib.pyplot as plt
import os

imput_directory = "input_data"
result_directory = "results"
look_angle = []
azimuth_angle = []


# Get metadata files
metadata_files = bs.get_xml_files(imput_directory)

# Extract look and azimuth angles
for i in range(len(metadata_files)):
    look, azimuth = bs.extract_angles(metadata_files[i])
    look_angle.append(look)
    azimuth_angle.append(azimuth)

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
