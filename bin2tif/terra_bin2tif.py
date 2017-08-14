#!/usr/bin/env python

"""
This extractor triggers when a file is added to a dataset in Clowder.

It checks for _left and _right BIN files to convert them into
JPG and TIF formats.
 """

import os
import logging
import shutil

from pyclowder.utils import CheckMessage
from pyclowder.files import upload_to_dataset
from pyclowder.datasets import download_metadata, upload_metadata, remove_metadata
from terrautils.metadata import get_extractor_metadata, get_terraref_metadata
from terrautils.extractors import TerrarefExtractor, is_latest_file, load_json_file, \
    create_geotiff, create_image, calculate_gps_bounds, build_metadata, build_dataset_hierarchy

import bin_to_geotiff as bin2tiff


class StereoBin2JpgTiff(TerrarefExtractor):
    def __init__(self):
        super(StereoBin2JpgTiff, self).__init__()

        # parse command line and load default logging configuration
        self.setup(sensor='stereoTop_geotiff')

    def check_message(self, connector, host, secret_key, resource, parameters):
        if not is_latest_file(resource):
            return CheckMessage.ignore

        # Check for a left and right BIN file - skip if not found
        found_left = False
        found_right = False
        for f in resource['files']:
            if 'filename' in f:
                if f['filename'].endswith('_left.bin'):
                    found_left = True
                elif f['filename'].endswith('_right.bin'):
                    found_right = True
        if not (found_left and found_right):
            return CheckMessage.ignore

        # Check if outputs already exist unless overwrite is forced - skip if found
        if not self.force_overwrite:
            timestamp = resource['dataset_info']['name'].split(" - ")[1]
            lbase = self.sensors.get_sensor_path(timestamp, opts=['left'], ext='')
            rbase = self.sensors.get_sensor_path(timestamp, opts=['right'], ext='')
            out_dir = os.path.dirname(lbase)
            if (os.path.isfile(lbase+'jpg') and os.path.isfile(rbase+'jpg') and
                    os.path.isfile(lbase+'tif') and os.path.isfile(rbase+'tif')):
                logging.info("skipping dataset %s; outputs found in %s" % (resource['id'], out_dir))
                return CheckMessage.ignore

        # Check metadata to verify we have what we need
        md = download_metadata(connector, host, secret_key, resource['id'])
        if get_extractor_metadata(md, self.extractor_info['name']) and not self.force_overwrite:
            logging.info("skipping dataset %s; metadata indicates it was already processed" % resource['id'])
            return CheckMessage.ignore
        if get_terraref_metadata(md) and found_left and found_right:
            return CheckMessage.download
        return CheckMessage.ignore

    def process_message(self, connector, host, secret_key, resource, parameters):
        self.start_message()

        # Get left/right files and metadata
        img_left, img_right, metadata = None, None, None
        for fname in resource['local_paths']:
            if fname.endswith('_dataset_metadata.json'):
                all_dsmd = load_json_file(fname)
                # TODO: Remove this lowercase requirement for downstream
                metadata = bin2tiff.lower_keys(get_extractor_metadata(all_dsmd))
            elif fname.endswith('_left.bin'):
                img_left = fname
            elif fname.endswith('_right.bin'):
                img_right = fname
        if None in [img_left, img_right, metadata]:
            raise ValueError("could not locate each of left+right+metadata in processing")

        # Determine output location & filenames
        timestamp = resource['dataset_info']['name'].split(" - ")[1]
        lbase = self.sensors.create_sensor_path(timestamp, opts=['left'], ext='')
        rbase = self.sensors.create_sensor_path(timestamp, opts=['right'], ext='')
        out_dir = os.path.dirname(lbase)
        self.sensors.create_sensor_path(out_dir)

        left_jpg = lbase+'.jpg'
        right_jpg = rbase+'.jpg'
        left_tiff = lbase+'.tif'
        right_tiff = rbase+'.tif'
        uploaded_file_ids = []

        logging.info("...determining image shapes")
        left_shape = bin2tiff.get_image_shape(metadata, 'left')
        right_shape = bin2tiff.get_image_shape(metadata, 'right')
        (left_gps_bounds, right_gps_bounds) = calculate_gps_bounds(metadata)
        out_tmp_tiff = "/home/extractor/"+resource['dataset_info']['name']+".tif"

        # TODO: Store root collection name in sensors.py?
        target_dsid = build_dataset_hierarchy(connector, host, secret_key, self.clowderspace,
                                              self.sensors.get_display_name(), timestamp[:4], timestamp[:7],
                                              timestamp[:10], leaf_ds_name=resource['dataset_info']['name'])

        skipped_jpg = False
        if (not os.path.isfile(left_jpg)) or self.force_overwrite:
            logging.info("...creating & uploading left JPG")
            left_image = bin2tiff.process_image(left_shape, img_left, None)
            create_image(left_image, left_jpg)
            # Only upload the newly generated file to Clowder if it isn't already in dataset
            if left_jpg not in resource['local_paths']:
                fileid = upload_to_dataset(connector, host, secret_key, target_dsid, left_jpg)
                uploaded_file_ids.append(fileid)
            self.created += 1
            self.bytes += os.path.getsize(left_jpg)
        else:
            skipped_jpg = True

        if (not os.path.isfile(left_tiff)) or self.force_overwrite:
            logging.info("...creating & uploading left geoTIFF")
            if skipped_jpg:
                left_image = bin2tiff.process_image(left_shape, img_left, None)
            # Rename output.tif after creation to avoid long path errors
            create_geotiff(left_image, left_gps_bounds, out_tmp_tiff)
            shutil.move(out_tmp_tiff, left_tiff)
            if left_tiff not in resource['local_paths']:
                fileid = upload_to_dataset(connector, host, secret_key, target_dsid, left_tiff)
                uploaded_file_ids.append(fileid)
            self.created += 1
            self.bytes += os.path.getsize(left_tiff)
        del left_image

        skipped_jpg = False
        if (not os.path.isfile(right_jpg)) or self.force_overwrite:
            logging.info("...creating & uploading right JPG")
            right_image = bin2tiff.process_image(right_shape, img_right, None)
            create_image(right_image, right_jpg)
            if right_jpg not in resource['local_paths']:
                fileid = upload_to_dataset(connector, host, secret_key, target_dsid, right_jpg)
                uploaded_file_ids.append(fileid)
            self.created += 1
            self.bytes += os.path.getsize(right_jpg)
        else:
            skipped_jpg = True

        if (not os.path.isfile(right_tiff)) or self.force_overwrite:
            logging.info("...creating & uploading right geoTIFF")
            if skipped_jpg:
                right_image = bin2tiff.process_image(right_shape, img_right, None)
            create_geotiff(right_image, right_gps_bounds, out_tmp_tiff)
            shutil.move(out_tmp_tiff, right_tiff)
            if right_tiff not in resource['local_paths']:
                fileid = upload_to_dataset(connector, host, secret_key, target_dsid,right_tiff)
                uploaded_file_ids.append(fileid)
            self.created += 1
            self.bytes += os.path.getsize(right_tiff)
        del right_image

        # Tell Clowder this is completed so subsequent file updates don't daisy-chain
        metadata = build_metadata(host, self.extractor_info['name'], target_dsid, {
                "files_created": uploaded_file_ids
            }, 'dataset')
        upload_metadata(connector, host, secret_key, target_dsid, metadata)

        self.end_message()

if __name__ == "__main__":
    extractor = StereoBin2JpgTiff()
    extractor.start()
