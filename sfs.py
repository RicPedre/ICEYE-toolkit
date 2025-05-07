import rasterio
import numpy as np
import baseline_comparison as bs
import matplotlib.pyplot as plt
import numexpr as ne

# input files
dem_file = "input_data/alletch_tip_UTM.tif"
satellite_file = "input_data/ICEYE_X2_SLC_SLEA_1000466_20220904T093630.xml"


# read the DEM file
with rasterio.open(dem_file) as src:
    print("reading dem file")
    dem = src.read(1).astype(float)
    transform = src.transform
    crs = src.crs

# get the coordinates
print("extracting coordinates")
rows, cols = dem.shape
x_coords = np.arange(cols) * transform.a + transform.c
y_coords = np.arange(rows) * transform.e + transform.f
X, Y = np.meshgrid(x_coords, y_coords)
Z = dem

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


sate_vectors = bs.extract_state_vectors(satellite_file)
satellite_position = sate_vectors[0][len(sate_vectors[0]) // 2]

# Build 3D grid of points from DEM
print("building 3D grid of points")
points = np.dstack((X, Y, Z))

# Vector from satellite to ground points
print("computing look vector for exch point")
look_vec = points - satellite_position.reshape(1, 1, 3)

# Normalize
print("normalizing look vector")
look_vec /= np.linalg.norm(look_vec, axis=2, keepdims=True)


def lambertian(normals, look_vec):
    print("computing lambertian reflectance")
    # Lambertian reflectance
    dot = np.sum(normals * look_vec, axis=2)  # No reshape needed
    reflectance = np.clip(dot, 0, 1)
    return reflectance


normals = compute_normals(X, Y, Z)
reflectance = lambertian(normals, look_vec)

plt.imshow(reflectance, cmap="gray")
plt.title("Lambertian Reflectance")
plt.colorbar(label="Reflectance")
plt.savefig("results/lambertian_reflectance.png")
plt.show()
