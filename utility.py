import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import CRS
import numpy as np


class dem:
    def __init__(
        self,
        dem_data: np.ndarray = None,
        transform: rasterio.transform.Affine = None,
        crs: CRS = None,
        bounds: rasterio.coords.BoundingBox = None,
        width: int = None,
        height: int = None,
        vertical_datum: str = None,
        dem_file: str = None,
    ):
        """
        Initialize the DEM class.

        Parameters:
        dem_data: numpy.ndarray
            DEM data as a NumPy array.
        transform: rasterio.transform.Affine
            Affine transformation for the DEM.
        crs: pyproj.CRS
            Coordinate Reference System of the DEM.
        bounds: rasterio.coords.BoundingBox
            Bounds of the DEM.
        width: int
            Width of the DEM.
        height: int
            Height of the DEM.
        vertical_datum: str
            Vertical datum of the DEM.
        dem_file: str
            Path to the DEM file.
        """

        self.dem_data = dem_data
        self.transform = transform
        self.crs = crs
        self.bounds = bounds
        self.width = width
        self.height = height
        self.vertical_datum = vertical_datum
        self.dem_file = dem_file

        # If dem_data is provided, set the attributes accordingly
        if dem_data is not None and transform is not None and bounds is None:
            # Calculate bounds from DEM data, CRS, and transform
            self.bounds = rasterio.transform.array_bounds(
                dem_data.shape[0], dem_data.shape[1], transform
            )
            self.width = dem_data.shape[1]
            self.height = dem_data.shape[0]

        if dem_data is not None and bounds is not None and transform is None:
            if width is None:
                self.width = bounds.right - bounds.left
            if height is None:
                self.height = bounds.top - bounds.bottom
            # Calculate transform from DEM data, CRS, and bounds
            self.transform = rasterio.transform.from_bounds(
                self.bounds.left,
                self.bounds.bottom,
                self.bounds.right,
                self.bounds.top,
                self.width,
                self.height,
            )

    @classmethod
    def from_file(cls, dem_file) -> "dem":
        """
        Create a DEM object from a file.

        Parameters:
        dem_file (str): Path to the DEM file.

        Returns:
        dem: An instance of the DEM class.
        """
        with rasterio.open(dem_file) as src:
            dem_data = src.read(1).astype(float)
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            width = src.width
            height = src.height
            vertical_datum = None
            print(f"Reading DEM file: {dem_file}")
            dem_data = src.read(1).astype(float)

            if crs.to_dict().get("vertical_datum"):
                vertical_datum = crs.to_dict()["vertical_datum"]

        return cls(
            dem_data,
            transform,
            crs,
            bounds=bounds,
            width=width,
            height=height,
            vertical_datum=vertical_datum,
            dem_file=dem_file,
        )

    def info(self):
        """
        Print information about the DEM object.
        """
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

    def wgs2utm(self) -> "dem":
        """
        Transform a DEM file from WGS84 (also with vertical datum) to the appropriate UTM CRS.


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
        dst_dem = dem(
            dem_data=dem_utm,
            transform=dst_transform,
            crs=utm_crs,
            bounds=rasterio.transform.array_bounds(height, width, dst_transform),
            width=width,
            height=height,
            vertical_datum=self.vertical_datum,
        )

        # Return the transformed DEM, UTM CRS, transform, and vertical datum if present
        return dst_dem

    def save(self, output_file: str):
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

    def null_threshold(self, threshold: float) -> "dem":
        """
        Transform very low numbers in a DEM into null cells (NaN).

        Parameters:
            threshold (float): The threshold below which values are set to NaN.

        Returns:
            dem: A new DEM object with values below the threshold set to NaN.
        """
        # Replace values below the threshold with NaN
        dem_with_nulls = dem(
            dem_data=np.where(self.dem_data < threshold, np.nan, self.dem_data),
            transform=self.transform,
            crs=self.crs,
            bounds=self.bounds,
            width=self.width,
            height=self.height,
            vertical_datum=self.vertical_datum,
            dem_file=self.dem_file,
        )

        return dem_with_nulls
