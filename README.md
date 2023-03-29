#### BigEarthNet Reader

This script reads BigEarthNet image patches and their labels from Sentinel-1 and Sentinel-2 data and converts them into TensorFlow Record (TFRecord) format. This facilitates efficient reading and processing of the dataset for machine learning tasks.
#### Requirements

    Python 3.x
    TensorFlow
    One of the following packages to read GeoTIFF files:
        GDAL
        rasterio

#### Usage

```bash
python bigearthnet_reader.py -r1 <root_folder_s1> -r2 <root_folder_s2> -o <output_folder> [options]
```
    -r1, --root_folder_s1: Root folder path containing multiple patch folders of BigEarthNet-S1.
    -r2, --root_folder_s2: Root folder path containing multiple patch folders of BigEarthNet-S2.
    -o, --output_folder: Output folder for the generated TFRecords. Default is "tfrecords".
    -p1, --patch_folder: Use if you want to read a single patch (will be ignored if -r1 and -r2 are specified).
    -c, --cores: Number of cores to use for processing. Default is single process.
    -n, --num_examples: Limit the number of examples to read (not the number of patches; some patches may be skipped unless -b is used).
    -b, --bad_patches: Include bad patches (with cloud and shadow or seasonal snow) in the output.

#### Output

The generated TFRecords will be saved in the specified output folder, divided into three subfolders: "train", "val", and "test".
