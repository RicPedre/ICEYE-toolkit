import rasterio
import numpy as np
import baseline_comparison as bs
import matplotlib.pyplot as plt
import numexpr as ne
from utility import dem

# compute surface normals


def compute_normals(X, Y, Z):
    print("computing normals")
    dzdx = np.gradient(Z, axis=1)
    dzdy = np.gradient(Z, axis=0)
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(Z)
    norm = ne.evaluate("sqrt(nx**2 + ny**2 + nz**2)")
    return np.dstack((nx, ny, nz)) / norm[..., np.newaxis]


def compute_look_vector(satellite_position, X, Y, Z):
    # Build 3D grid of points from DEM
    print("building 3D grid of points")
    points = np.dstack((X, Y, Z))

    # Vector from satellite to ground points
    print("computing look vector for exch point")
    look_vec = points - satellite_position.reshape(1, 1, 3)

    # Normalize
    print("normalizing look vector")
    look_vec /= np.linalg.norm(look_vec, axis=2, keepdims=True)


def look_vector_from_angles(azimuth_angle, look_angle):
    az = np.deg2rad(azimuth_angle)
    el = np.deg2rad(90 - look_angle)  # elevation = 90 - off_nadir

    x = np.sin(el) * np.sin(az)
    y = np.sin(el) * np.cos(az)
    z = np.cos(el)
    print(f"look vector: {x}, {y}, {z}")
    return np.array([x, y, -z])  # assuming sensor looks downwards


def lambertian(normals, look_vec):
    print("computing lambertian reflectance")
    # Lambertian reflectance
    dot = np.sum(normals * look_vec, axis=2)  # No reshape needed
    reflectance = np.clip(dot, 0, 1)
    return reflectance


# input files
dem_file = "input_data/alletch_tip_UTM.tif"
satellite_file = "input_data/ICEYE_X11_SLC_SLEA_1000478_20220907T012034.xml"

alletch = dem.from_file(dem_file)

# get the coordinates
print("extracting coordinates and creating meshgrid")
rows, cols = alletch.dem_data.shape
x_coords = np.arange(cols) * alletch.transform.a + alletch.transform.c
y_coords = np.arange(rows) * alletch.transform.e + alletch.transform.f
X, Y = np.meshgrid(x_coords, y_coords)
Z = alletch.dem_data

sate_vectors = bs.extract_state_vectors(satellite_file)
look_angle, azimuth_angle = bs.extract_angles(satellite_file)
satellite_position = sate_vectors[0][len(sate_vectors[0]) // 2]

look_vec = look_vector_from_angles(azimuth_angle, look_angle)

normals = compute_normals(X, Y, Z)

reflectance = lambertian(normals, look_vec)

plt.imshow(reflectance, cmap="gray")
plt.title("Lambertian Reflectance")
plt.colorbar(label="Reflectance")
plt.savefig("results/lambertian_reflectance.png")
plt.show()
