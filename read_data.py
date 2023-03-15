#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script reads GeoTIFF files each of which is for one spectral
# band of a Sentinel-21 image patch in the BigEarthNet Archive.
#
# The script is capable of reading either  all spectral bands of one patch
# folder (-p option) or all bands for all patches (-r option).
#
# After reading files, Sentinel-1 image patch values can be used as numpy array
# for further purposes.
#
# read_patch --help can be used to learn how to use this script.
#
# Date: 22 Dec 2020
# Version: 1.0.1
# Usage: read_patch.py [-h] [-p PATCH_FOLDER] [-r ROOT_FOLDER]

from __future__ import print_function
import argparse
import os
import json
import tensorflow as tf
import numpy as np
import tqdm
from multiprocessing import Pool


def parse_args():
    parser = argparse.ArgumentParser(
        description="This script reads the BigEarthNet image patches"
    )
    parser.add_argument(
        "-p1",
        "--patch_folder",
        dest="patch_folder",
        help="use if you want to read a sinlg patch",
    )
    parser.add_argument(
        "-r1",
        "--root_folder_s1",
        dest="root_folder_s1",
        help="root folder path contains multiple patch folders of BigEarthNet-S1",
    )
    parser.add_argument(
        "-r2",
        "--root_folder_s2",
        dest="root_folder_s2",
        help="root folder path contains multiple patch folders of BigEarthNet-S2",
    )
    parser.add_argument(
        "-c",
        "--cores",
        dest="cores",
        type=int,
        default=1,
        help="number of cores to use, default is single process"
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        dest="output_folder",
        default="tfrecords",
    )

    parser.add_argument(
        "-n",
        "--num_examples",
        type=int,
        dest="num_examples",
        help="Limit the number of examples to read (note this is not the number of patches some are skipped unless -b is used)",
    )

    parser.add_argument(
        "-b",
        "--bad_patches",
        dest = "bad_patches",
        help="Include bad patches in the output",
    )

    args = parser.parse_args()
    return args

LABELS_19 = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

# fmt: off
LABELS = [
    'Agro-forestry areas', 'Airports',
    'Annual crops associated with permanent crops', 'Bare rock',
    'Beaches, dunes, sands', 'Broad-leaved forest', 'Burnt areas',
    'Coastal lagoons', 'Complex cultivation patterns', 'Coniferous forest',
    'Construction sites', 'Continuous urban fabric',
    'Discontinuous urban fabric', 'Dump sites', 'Estuaries',
    'Fruit trees and berry plantations', 'Green urban areas',
    'Industrial or commercial units', 'Inland marshes', 'Intertidal flats',
    'Land principally occupied by agriculture, with significant areas of '
    'natural vegetation', 'Mineral extraction sites', 'Mixed forest',
    'Moors and heathland', 'Natural grassland', 'Non-irrigated arable land',
    'Olive groves', 'Pastures', 'Peatbogs', 'Permanently irrigated land',
    'Port areas', 'Rice fields', 'Road and rail networks and associated land',
    'Salines', 'Salt marshes', 'Sclerophyllous vegetation', 'Sea and ocean',
    'Sparsely vegetated areas', 'Sport and leisure facilities',
    'Transitional woodland/shrub', 'Vineyards', 'Water bodies', 'Water courses'
]
#fmt: on

GROUP_LABELS = {
    'Continuous urban fabric': 'Urban fabric',
    'Discontinuous urban fabric': 'Urban fabric',
    'Non-irrigated arable land': 'Arable land',
    'Permanently irrigated land': 'Arable land',
    'Rice fields': 'Arable land',
    'Vineyards': 'Permanent crops',
    'Fruit trees and berry plantations': 'Permanent crops',
    'Olive groves': 'Permanent crops',
    'Annual crops associated with permanent crops': 'Permanent crops',
    'Natural grassland': 'Natural grassland and sparsely vegetated areas',
    'Sparsely vegetated areas': 'Natural grassland and sparsely vegetated areas',
    'Moors and heathland': 'Moors, heathland and sclerophyllous vegetation',
    'Sclerophyllous vegetation': 'Moors, heathland and sclerophyllous vegetation',
    'Inland marshes': 'Inland wetlands',
    'Peatbogs': 'Inland wetlands',
    'Salt marshes': 'Coastal wetlands',
    'Salines': 'Coastal wetlands',
    'Water bodies': 'Inland waters',
    'Water courses': 'Inland waters',
    'Coastal lagoons': 'Marine waters',
    'Estuaries': 'Marine waters',
    'Sea and ocean': 'Marine waters'
}

# radar and spectral band names to read related GeoTIFF files
BAND_NAMES_S1 = ["VV", "VH"]
BAND_NAMES_S2 = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]
RGB_BANDS = ['B04', 'B03', 'B02']

def label_mapping_19(label_list):
    # From https://bigearth.eu/BigEarthNetListofClasses.pdf
    # 19 classes
    out_labels_19 = []
    for label in label_list:
        if label in LABELS_19:
            out_labels_19.append(label)
            continue
        if label not in GROUP_LABELS:
            continue
        label_19 = GROUP_LABELS[label] 
        if label_19 in out_labels_19:
            continue
        out_labels_19.append(GROUP_LABELS[label])
    return out_labels_19
        

def read_bands(patch_folder_path, patch_name, band_names):
    band_dict = {}
    for band_name in band_names:
        # First finds related GeoTIFF path and reads values as an array
        band_path = os.path.join(
            patch_folder_path, patch_name + "_" + band_name + ".tif"
        )
        if gdal_existed:
            band_ds = gdal.Open(band_path, gdal.GA_ReadOnly)
            raster_band = band_ds.GetRasterBand(1)
            band_data = raster_band.ReadAsArray()
        elif rasterio_existed:
            band_ds = rasterio.open(band_path)
            band_data = band_ds.read(1)
        # band_data keeps the values of band band_name for the patch patch_name
        # print(
        #     "INFO: band",
        #     band_name,
        #     "of patch",
        #     patch_name,
        #     "is ready with size",
        #     band_data.shape,
        # )
        band_dict[band_name] = band_data

    return band_dict

def multi_hot_encode(patch_labels, all_labels):
    multi_hot = np.zeros(len(all_labels), dtype=int)
    for label in patch_labels:
        label_index = all_labels.index(label)
        multi_hot[label_index] = 1
    return multi_hot

        
if __name__ == "__main__":
    args = parse_args()    
    os.makedirs(args.output_folder, exist_ok=True)

    # Checks the existence of patch folders and populate the list of patch folder paths
    folder_path_list = []
    if args.root_folder_s1 and args.root_folder_s2:
        print("INFO: -p argument will not be considered since -r argument is defined")
        if not os.path.exists(args.root_folder_s1):
            print("ERROR: folder", args.root_folder_s1, "does not exist")
            exit(1)
        elif not os.path.exists(args.root_folder_s2):
            print("ERROR: folder", args.root_folder_s2, "does not exist")
            exit(1)

        if args.num_examples is not None:
            assert args.num_examples > 1, "num_examples should be greater than 1"
        for idx, patch_dir in tqdm.tqdm(enumerate(os.listdir(args.root_folder_s1))):
            folder_path_list.append(os.path.join(args.root_folder_s1, patch_dir))
            if args.num_examples is not None:
                if idx > args.num_examples:
                    break
        # Skip first this is the name of the root folder
        folder_path_list = folder_path_list[1:-1]

        if len(folder_path_list) == 0:
            print("ERROR: there is no patch directories in the root folder")
            exit(1)

    elif not args.patch_folder_s1:
        print("ERROR: at least one of -p and -r arguments is required")
        exit(1)
    else:
        if not os.path.exists(args.patch_folder_s1):
            print("ERROR: folder", args.patch_folder, "does not exist")
            exit(1)
        folder_path_list = [args.patch_folder_s1]

    # Checks the existence of required python packages
    gdal_existed = rasterio_existed = georasters_existed = False
    try:
        import gdal

        gdal_existed = True
        print("INFO: GDAL package will be used to read GeoTIFF files")
    except ImportError:
        try:
            import rasterio

            rasterio_existed = True
            print("INFO: rasterio package will be used to read GeoTIFF files")
        except ImportError:
            print(
                "ERROR: please install either GDAL or rasterio package to read GeoTIFF files"
            )
            exit(1)

    # Reads spectral bands of all patches whose folder names are populated before
    train_patches = []
    val_patches = []
    test_patches = []
    bad_patches = []

    if args.bad_patches is not None:
        with open("patches_with_cloud_and_shadow.csv", "r") as f:
            cloud_and_shadow = f.readlines()
            cloud_and_shadow = [x.strip() for x in cloud_and_shadow]
            bad_patches.extend(cloud_and_shadow)

        with open("patches_with_seasonal_snow.csv", "r") as f:
            seasonal_snow = f.readlines()
            seasonal_snow = [x.strip() for x in seasonal_snow]
            bad_patches.extend(seasonal_snow)
    
    with open("bigearthnet-train.txt", "r") as f:
        train = f.readlines()
        train = [x.strip() for x in train]

    with open("bigearthnet-val.txt", "r") as f:
        val = f.readlines()
        val = [x.strip() for x in val]
    
    with open("bigearthnet-test.txt", "r") as f:
        test = f.readlines()
        test = [x.strip() for x in test]
    
    os.makedirs(os.path.join(args.output_folder, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "val"), exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "test"), exist_ok=True)

    def create_record(patch_folder_path):
        patch_folder_path_s1 = os.path.realpath(patch_folder_path)
        patch_name_s1 = os.path.basename(patch_folder_path)
        labels_metadata_path_s1 = os.path.join(
            patch_folder_path_s1, patch_name_s1 + "_labels_metadata.json"
        )

        # Reads labels_metadata json file
        with open(labels_metadata_path_s1, "r") as f:
            labels_metadata_s1 = json.load(f)

        # get corresponding s2 patch
        patch_name_s2 = labels_metadata_s1["corresponding_s2_patch"]
        patch_folder_path_s2 = os.path.join(args.root_folder_s2, patch_name_s2)
        if os.path.basename(patch_folder_path_s2) in bad_patches and args.bad_patches is not None:
            print("INFO: skipping patch", patch_name_s2)
            return

        # read S1 bands
        s1_bands = read_bands(patch_folder_path_s1, patch_name_s1, BAND_NAMES_S1)
        # read corresponding S2 bands
        s2_bands = read_bands(patch_folder_path_s2, patch_name_s2, BAND_NAMES_S2)

        features_dict = {}
        for feature_name, value in s1_bands.items():
            features_dict[feature_name] = tf.train.Feature(
                float_list=tf.train.FloatList(value=value.reshape(-1))
            )

        for feature_name, value in s2_bands.items():
            features_dict[feature_name] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=value.reshape(-1))
            )
        
        labels =  labels_metadata_s1["labels"]
        labels_19 = label_mapping_19(labels_metadata_s1["labels"])
        multi_hot_43 = multi_hot_encode(labels, LABELS)
        multi_hot_19 = multi_hot_encode(labels_19, LABELS_19)

        features_dict["BigEarthNet-43_labels"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[label.encode("utf-8") for label in labels])
        )
        features_dict["BigEarthNet-19_labels"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[label.encode("utf-8") for label in labels_19])
        )
        features_dict["BigEarthNet-43_labels_multi_hot"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=multi_hot_43)
        )
        features_dict["BigEarthNet-19_labels_multi_hot"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=multi_hot_19)
        )
        features_dict["patch_name_s1"] = tf.train.Feature( 
            bytes_list=tf.train.BytesList(value=[patch_name_s1.encode("utf-8")])
        )
        features_dict["patch_name_s2"] = tf.train.Feature( 
            bytes_list=tf.train.BytesList(value=[patch_name_s2.encode("utf-8")])
        )

        out_example = tf.train.Example(features=tf.train.Features(feature=features_dict))
        out_name = patch_name_s2 + ".tfrecord"
        if patch_name_s2 in train: 
            out_path = os.path.join(args.output_folder, "train", out_name) 
        elif patch_name_s2 in val: 
            out_path = os.path.join(args.output_folder, "val", out_name) 
        elif patch_name_s2 in test: 
            out_path = os.path.join(args.output_folder, "test", out_name) 
        else:
            raise("Unexpected patch name {}".format(patch_name_s2))
        
        with tf.io.TFRecordWriter(out_path) as writer:
            writer.write(out_example.SerializeToString())

    pool = Pool(args.cores)
    for _ in tqdm.tqdm(pool.imap_unordered(create_record, folder_path_list), total=len(folder_path_list)):
        ...
        




    # print(multi_hot_19)
        # print(features_dict["19_labels_multi_hot"])




        

