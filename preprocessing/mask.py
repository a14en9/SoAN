"""
Generate Masks from Tiffs / Shapefiles

This module has utilities for converting raw geotiffs into numpy arrays that
can be used for subsequent training.
"""
import argparse
import os
import pathlib
import warnings
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import box, Polygon
from shapely.ops import cascaded_union
import geopandas as gpd
import pandas as pd
import rasterio
import yaml

warnings.simplefilter(action="ignore", category=FutureWarning)


def generate_masks(img_paths, shps_paths, border_paths=[], output_base="mask",
                   out_dir=None):
    """A wrapper of generate_mask, to make labels for each input

    Args:
        image_paths(List): A list of Strings of the paths to the raw images
        shps_paths(List): A list of Strings of the paths to the raw polygons
        border_paths(List): A list of Strings of the paths to the border polygon
        output_base(String): The basenames for all the output numpy files
        out_dir(String): The directory to which all the results are saved
    Returns:
        Writes a csv to metadata path
    """
    if not out_dir:
        out_dir = pathlib.Path("processed", "masks")

    pathlib.Path(out_dir).mkdir(parents=True)
    cols = ["id", "img", "mask", "border",
            "img_width", "img_height", "mask_width", "mask_height"]
    metadata = pd.DataFrame({k: [] for k in cols})
    metadata_path = pathlib.Path(out_dir, "mask_metadata.csv")
    if not metadata_path.exists():
        metadata.to_csv(metadata_path, index=False)
    else:
        raise ValueError(f"Cannot overwrite {metadata_path}.")

    for k, img_path in enumerate(img_paths):
        print(f"working on image {k + 1} / {len(img_paths)}")
        img, shps = rasterio.open(img_path), []
        for path in shps_paths[k]:
            gdf = gpd.read_file(path)
            gdf_crs = rasterio.crs.CRS.from_string(gdf.crs.to_string())
            if gdf_crs != img.meta["crs"]:
                gdf = gdf.to_crs(img.meta["crs"].data)
            shps.append(gdf)

        # build mask over tiff's extent, and save
        shps = clip_shapefile(img.bounds, img.meta, shps)
        mask = generate_mask(img.meta, shps)
        out_path = pathlib.Path(out_dir, f"{output_base}_{k:02}")
        np.save(str(out_path), mask)
        # get borders
        if border_paths:
            border_mask = get_border_mask(img, border_paths[k])
            border_path = pathlib.Path(out_dir, f"border_{k:02}.npy")
            np.save(str(border_path), border_mask)
        else:
            border_path = None

        pd.DataFrame({
                "img_path": img_path,
                "mask": str(out_path) + ".npy",
                "border": str(border_path),
                "width": img.meta["width"],
                "height": img.meta["height"],
                "mask_width": mask.shape[1],
                "mask_height": mask.shape[0],
            },
            index=[k],
        ).to_csv(metadata_path, header=False, mode="a")

def get_border_mask(img, border_path):
    """Get mask of a border"""
    #TODO: Use one function for any mask
    gdf = gpd.read_file(border_path)
    gdf_crs = rasterio.crs.CRS.from_string(gdf.crs.to_string())
    if gdf_crs != img.meta["crs"]:
        gdf = gdf.to_crs(img.meta["crs"].data)
    gdf = clip_shapefile(img.bounds, img.meta, [gdf])[0]
    mask = generate_mask(img.meta, [gdf])

    return mask

def check_crs(crs_a, crs_b):
    """Verify that two CRS objects Match

    :param crs_a: The first CRS to compare.
    :type crs_a: rasterio.crs
    :param crs_b: The second CRS to compare.
    :type crs_b: rasterio.crs
    :side-effects: Raises an error if the CRS's don't agree
    """
    if rasterio.crs.CRS.from_string(crs_a.to_string()) != rasterio.crs.CRS.from_string(
            crs_b.to_string()
    ):
        raise ValueError("Coordinate reference systems do not agree")


def generate_mask(img_meta, shps):
    """Generate K-Channel Label Masks over Raster Image

    :param img_meta: The metadata field associated with a geotiff. Expected to
      contain transform (coordinate system), height, and width fields.
    :type img_meta: rasterio.metadata
    :param shps: A list of K geopandas shapefiles, used to build the mask.
      Assumed to be in the same coordinate system as img_data.
    :type: [gpd.GeoDataFrame]
    :return mask: A K channel binary numpy array. The k^th channel gives the
      binary mask for the k^th input shapefile.
    """
    result = np.zeros((img_meta["height"], img_meta["width"], len(shps)), dtype="uint8")
    for k, shp in enumerate(shps):
        check_crs(img_meta["crs"], shp.crs)
        result[:, :, k] = channel_mask(img_meta, shp)

    return result


def channel_mask(img_meta, shp):
    """Generate 1-channel label mask over raster Image

    Args:
      img_meta (rasterio.metadata): The metadata associated with the location
        on which to build the mask.
      shp (gpd.GeoDataFrame): A geopandas shapefile, used to build the mask.
    """
    poly_shp = []
    for _, row in shp.iterrows():
        if row["geometry"].geom_type == "Polygon":
            poly_shp += [poly_from_coord(row["geometry"], img_meta["transform"])]

        else:  # case if multipolygon
            for geom in row["geometry"]:
                poly_shp += [poly_from_coord(geom, img_meta["transform"])]

    im_size = (img_meta["height"], img_meta["width"])

    try:
        result = rasterize(shapes=poly_shp, out_shape=im_size)
    except ValueError as e:
        if str(e) == 'No valid geometry objects found for rasterize':
            result = np.zeros(im_size)
        else: raise

    return result


def poly_from_coord(polygon, transform):
    """Get a transformed polygon
    https://lpsmlgeo.github.io/2019-09-22-binary_mask/
    """
    poly_pts = []
    poly = cascaded_union(polygon)
    for i in np.array(poly.exterior.coords):
        poly_pts.append(~transform * tuple(i)[:2])  # in case polygonz format
    return Polygon(poly_pts)


def clip_shapefile(img_bounds, img_meta, shps):
    """Clip Shapefile Extents to Image Bounding Box

    :param img_bounds: The rectangular lat/long bounding box associated with a
      raster tiff.
    :type img_bounds: Tuple
    :param img_meta: The metadata field associated with a geotiff. Expected to
      contain transform (coordinate system), height, and width fields.
    :type img_meta: rasterio.metadata
    :param shps: A list of K geopandas shapefiles, used to build the mask.
      Assumed to be in the same coordinate system as img_data.
    :type: [gpd.GeoDataFrame]
    :return result: The same shapefiles as shps, but with polygons that don't
      overlap the img bounding box removed.
    """
    bbox = box(*img_bounds)
    bbox_poly = gpd.GeoDataFrame(
        {"geometry": bbox}, index=[0], crs=img_meta["crs"].data
    )
    result = []
    for shp in shps:
        check_crs(img_meta["crs"], shp.crs)
        result.append(shp.loc[shp.intersects(bbox_poly["geometry"][0])])
    return result
def mask_generation(img_paths, shps_paths, output_base="mask",
                   out_dir=None):
    """A wrapper of generate_mask, to make labels for each input

    Args:
        image_paths(List): A list of Strings of the paths to the raw images
        shps_paths(List): A list of Strings of the paths to the raw polygons
        border_paths(List): A list of Strings of the paths to the border polygon
        output_base(String): The basenames for all the output numpy files
        out_dir(String): The directory to which all the results are saved
    Returns:
        Writes a csv to metadata path
    """
    if not out_dir:
        out_dir = str(pathlib.Path("processed", "masks"))

    pathlib.Path(out_dir).mkdir(parents=True)
    cols = ["id", "img", "mask", "img_width", "img_height", "mask_width", "mask_height"]
    metadata = pd.DataFrame({k: [] for k in cols})
    metadata_path = pathlib.Path(out_dir, "mask_metadata.csv")
    if not metadata_path.exists():
        metadata.to_csv(metadata_path, index=False)
    else:
        raise ValueError(f"Cannot overwrite {metadata_path}.")

    for k, img_path in enumerate(img_paths):
        print(f"working on image {k + 1} / {len(img_paths)}")
        img, imgshps = rasterio.open(img_path), []
        mask = gdal.Open(shps_paths[k])
        mask = np.array(mask.GetRasterBand(1).ReadAsArray())
        out_path = str(pathlib.Path(out_dir, f"{output_base}_{k:02}"))
        np.save(out_path, mask)

        pd.DataFrame({
                "img_path": img_path,
                "mask": str(out_path) + ".npy",
                "width": img.meta["width"],
                "height": img.meta["height"],
                "mask_width": mask.shape[1],
                "mask_height": mask.shape[0],
            },
            index=[k],
        ).to_csv(metadata_path, header=False, mode="a")
if __name__ == '__main__':
    from osgeo import gdal
    """
    #######  Extracting all filenames under the image folder and saving them into a text file
    import os
    with open("filename.txt", "w") as a:
        for files in os.walk(r'../../patches/EEImages-20210316T212054Z-001/EEImages'):
            for filename in files[2]:
                a.write(str(filename) + os.linesep)
    """
    data_dir = pathlib.Path("../../patches")
    img_dir = pathlib.Path(data_dir / "raster_data")
    mask_dir = pathlib.Path(data_dir / "mask_imgs")
    with open('filename.txt', 'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    img_dirs = []
    mask_dirs = []
    for i, name in enumerate(content):
        img_dirs.append(str(pathlib.Path(img_dir / name)))
        mask_dirs.append(str(pathlib.Path(mask_dir / name)))
    outputs = mask_generation(img_dirs, mask_dirs)
    print('finished generation!')