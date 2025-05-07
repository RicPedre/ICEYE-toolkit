import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS
import numpy as np


class dem:
    def __init__(self, dem_file):
        self.dem_file = dem_file
        self.dem_data = None

        with rasterio.open(self.dem_file) as src:
            self.transform = src.transform
            self.crs = src.crs
            self.bounds = src.bounds
            self.width = src.width
            self.height = src.height

        if self.crs.to_dict().get("vertical_datum"):
            self.vertical_datum = self.crs.to_dict()["vertical_datum"]

    def read(self):
        with rasterio.open(self.dem_file) as src:
            self.dem_data = src.read(1).astype(float)

    def display_info(self):
        print(f"DEM File: {self.dem_file}")
        print(f"CRS: {self.crs}")
        print(f"Bounds: {self.bounds}")
        print(f"Transform: {self.transform}")
        print(
            f"Vertical Datum: {self.vertical_datum if self.vertical_datum else 'None'}"
        )
        print(
            f"DEM Data Shape: {self.dem_data.shape if self.dem_data is not None else 'file not read yet. run read() first.'}"
        )

    def wgs2utm(self):
        """
        Transform a DEM file from WGS84 (with vertical datum) to the appropriate UTM CRS.

        Parameters:
        dem_file (str): Path to the DEM file.

        Returns:
        tuple: Transformed DEM data (NumPy array), UTM CRS, transform, and vertical datum (if present).
        """

        if self.dem_data is None:
            raise ValueError("DEM data not read yet. Please call read() method first.")

        # Ensure the DEM is in WGS84 or EPSG:9707
        if self.crs != CRS.from_epsg(4326) and self.crs != CRS.from_epsg(9707):
            raise ValueError(
                f"The DEM file is not in WGS84 or EPSG:9707. The DEM is {self.crs}."
            )

        # Calculate the center longitude of the DEM
        center_longitude = (self.bounds.left + self.bounds.right) / 2
        center_latitude = (self.bounds.top + self.bounds.bottom) / 2

        # Determine the UTM zone
        utm_zone = int((center_longitude + 180) / 6) + 1
        is_northern = (
            center_latitude >= 0
        )  # Check if the DEM is in the northern hemisphere

        # Define the UTM CRS
        if is_northern:
            utm_crs = CRS.from_epsg(32600 + utm_zone)  # Northern hemisphere
        else:
            utm_crs = CRS.from_epsg(32700 + utm_zone)  # Southern hemisphere

        print(f"Center Longitude: {center_longitude}")
        print(f"UTM Zone: {utm_zone}")
        print(f"UTM CRS: {utm_crs}")

        print("Transforming DEM to the correct UTM CRS...")
        # Calculate the transform and dimensions for the new UTM CRS

        dst_transform, width, height = calculate_default_transform(
            self.crs, utm_crs, self.width, self.height, *self.bounds
        )

        # Create an empty array for the reprojected DEM
        dem_utm = np.empty((height, width), dtype=np.float32)

        # Reproject the DEM to the UTM CRS
        reproject(
            source=self.dem_data,
            destination=dem_utm,
            src_transform=self.transform,
            src_crs=self.crs,
            dst_transform=dst_transform,
            dst_crs=utm_crs,
        )
        print("Transformation complete.")

        # update the self attributes with the new values
        self.transform = dst_transform
        self.crs = utm_crs
        self.bounds = rasterio.transform.array_bounds(height, width, dst_transform)
        self.width = width
        self.height = height
        self.dem_data = dem_utm
        # Return the transformed DEM, UTM CRS, transform, and vertical datum if present
        return dem_utm, utm_crs, dst_transform, self.vertical_datum

    def save(self, output_file):
        """
        Save the DEM data to a new file.

        Parameters:
        output_file (str): Path to the output file.
        """
        with rasterio.open(
            output_file,
            "w",
            driver="GTiff",
            height=self.height,
            width=self.width,
            count=1,
            dtype=self.dem_data.dtype,
            crs=self.crs,
            transform=self.transform,
        ) as dst:
            dst.write(self.dem_data, 1)
