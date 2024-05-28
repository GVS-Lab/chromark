import copy
import logging
import os
import shutil
from typing import List

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import tifffile
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from skimage import filters, morphology, segmentation, measure, exposure, color, io
from skimage.io import imread
from tqdm import tqdm

from src.utils.basic.io import get_file_list, save_figure_as_png
from src.utils.basic.segmentation import get_nuclear_mask_in_3d, pad_image
from src.utils.basic.visualization import plot_colored_3d_segmentation


class SegmentationPipeline(object):
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        file_type_filter: str = None,
        normalize_channels: bool = False,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.file_list = get_file_list(
            self.input_dir, file_type_filter=file_type_filter
        )
        self.normalize_channels = normalize_channels
        self.file_name = None
        self.raw_image = None

    def read_in_image(self, index: int):
        self.file_name = self.file_list[index]
        self.raw_image = tifffile.imread(self.file_name)
        if self.normalize_channels:
            self.apply_channel_normalization()

    def apply_channel_normalization(self, channels=None):
        if channels is None:
            channels = []
        normalized_raw_image = copy.deepcopy(self.raw_image)
        for c in channels:
            channel_img = self.raw_image[:, c, :, :].astype(np.float32)
            normalized_raw_image[:, c, :, :] = np.array(
                (
                    (channel_img - channel_img.min()) / (channel_img.max()
                    - channel_img.min())
                )
                * ((2 ** 16) - 1),
                dtype=np.uint16,
            )
        self.raw_image = normalized_raw_image


class NucleiSegmentationPipeline(SegmentationPipeline):
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        channels: List[str] = None,
        file_type_filter: str = None,
        normalize_channels: bool = False,
    ):
        super().__init__(
            input_dir=input_dir,
            output_dir=output_dir,
            file_type_filter=file_type_filter,
            normalize_channels=normalize_channels,
        )
        self.nuclear_crops_2d_dir = None
        self.nuclear_crops_3d_dir = None
        self.nuclei_ids = None
        self.nuclear_crop_labels = None
        self.nuclei_masks = None
        self.file_name = None
        self.raw_image = None
        self.dna_image = None
        self.z_projection = None
        self.processed_projection = None
        self.labeled_projection = None
        self.labeled_image = None
        self.segmentation_visualization = None
        self.nuclear_crops = None
        self.labeled_marker_projections = dict()
        self.channels = channels

    def read_in_image(self, index: int):
        super().read_in_image(index=index)
        if len(self.raw_image.shape) == 4:
            self.dna_image = self.raw_image[:, 0, :, :]
            self.z_projection = self.dna_image.max(axis=0)
            self.processed_projection = copy.deepcopy(self.z_projection)
            return True
        else:
            logging.critical(
                "Image at location {} could not be read - skipping.".format(
                    self.file_list[index]
                )
            )

    def apply_lumination_correction(self, gamma: float = 0.7):
        self.processed_projection = exposure.adjust_gamma(
            self.processed_projection, gamma
        )

    def apply_morphological_closing(self):
        self.processed_projection = morphology.closing(self.processed_projection)

    def apply_image_filter(self, filter_type: str, **kwargs):
        if filter_type == "median":
            self.processed_projection = filters.median(self.processed_projection)
        elif filter_type == "gaussian":
            self.processed_projection = filters.gaussian(
                self.processed_projection, *kwargs
            )
        elif filter_type == "otsu":
            threshold = filters.threshold_otsu(self.processed_projection)
            self.processed_projection = self.processed_projection > threshold
        else:
            raise NotImplementedError("Unknown filter type: {}".format(filter_type))

    def clean_binary_segmentation(self, min_object_size: int = 1000):
        self.processed_projection = segmentation.clear_border(self.processed_projection)
        self.processed_projection = morphology.remove_small_objects(
            ndi.binary_fill_holes(self.processed_projection), min_size=min_object_size
        )
        self.processed_projection = ndi.binary_fill_holes(self.processed_projection)

    def segment_distance_transform_watershed(self, distance_threshold: float = 0.5):
        d = ndi.distance_transform_edt(self.processed_projection)
        markers = morphology.binary_erosion(d > d.max() * distance_threshold)
        markers = measure.label(markers)
        self.labeled_projection = segmentation.watershed(
            -d, markers, mask=self.processed_projection
        )

    def segment_by_connected_components(self):
        self.labeled_projection = measure.label(self.processed_projection)

    def cleaned_labeled_segmentation(
        self, min_area: float = 0, max_area: float = 100000
    ):
        self.labeled_projection = morphology.remove_small_objects(
            self.labeled_projection, min_size=min_area
        )
        regions = measure.regionprops(
            self.labeled_projection, intensity_image=self.z_projection
        )
        for region in regions:
            if region.area > max_area:
                self.labeled_projection[self.labeled_projection == region.label] = 0
        self.labeled_projection = measure.label(self.labeled_projection > 0)

    def get_nuclear_crops(self, expansion: int = 1):
        self.nuclear_crops = []
        self.nuclear_crop_labels = []
        regions = measure.regionprops(
            self.labeled_projection, intensity_image=self.z_projection
        )
        for region in regions:

            xmin, ymin, xmax, ymax = region.bbox
            xmin = max(0, xmin - expansion)
            ymin = max(0, ymin - expansion)
            xmax = min(xmax + expansion, self.labeled_projection.shape[0])
            ymax = min(ymax + expansion, self.labeled_projection.shape[1])
            crop = np.array(self.raw_image[:, :, xmin:xmax, ymin:ymax])

            # Set all values outside of z_projected nuclear mask to 0
            convex_hull = region.convex_image.astype(int)
            padded_convex_hull = np.zeros_like(crop[0, 0])
            padded_convex_hull = pad_image(convex_hull, padded_convex_hull.shape)
            for i in range(expansion):
                padded_convex_hull = ndi.binary_dilation(padded_convex_hull)
            z_mask = np.zeros_like(crop)
            for i in range(len(z_mask)):
                for j in range(len(z_mask[0])):
                    z_mask[i, j] = padded_convex_hull
            masked_crop = crop * z_mask
            self.nuclear_crops.append(masked_crop)
            self.nuclear_crop_labels.append(region.label)

    def save_nuclear_crops(self):
        self.nuclei_ids = []
        self.nuclear_crops_3d_dir = os.path.join(self.output_dir, "nuclear_crops_3d")
        self.nuclear_crops_2d_dir = os.path.join(self.output_dir, "nuclear_crops_2d")
        if not os.path.exists(self.nuclear_crops_3d_dir):
            os.makedirs(self.nuclear_crops_3d_dir, exist_ok=True)
        if not os.path.exists(self.nuclear_crops_2d_dir):
            os.makedirs(self.nuclear_crops_2d_dir, exist_ok=True)
        nuclei_file_name = os.path.split(self.file_name)[1]
        nuclei_3d_file_name = os.path.join(self.nuclear_crops_3d_dir, nuclei_file_name)
        nuclei_2d_file_name = os.path.join(self.nuclear_crops_2d_dir, nuclei_file_name)
        nuclei_3d_file_name = nuclei_3d_file_name[: nuclei_3d_file_name.index(".")]
        nuclei_2d_file_name = nuclei_2d_file_name[: nuclei_2d_file_name.index(".")]

        for i in range(len(self.nuclear_crops)):
            nucleus_3d_file = nuclei_3d_file_name + "_{}".format(i) + ".tif"
            self.nuclei_ids.append(
                nuclei_file_name[: nuclei_file_name.index(".")] + "_{}".format(i)
            )
            nucleus_2d_file = (
                nuclei_2d_file_name + "_{}".format(self.nuclear_crop_labels[i]) + ".tif"
            )
            nucleus = self.nuclear_crops[i]
            nucleus_max_z = np.array(nucleus).max(axis=0)
            # ImageJ format requires TZCYXS
            nucleus = np.expand_dims(nucleus, 0)
            nucleus_max_z = np.expand_dims(nucleus_max_z, 0)
            nucleus_max_z = np.expand_dims(nucleus_max_z, 0)
            tifffile.imsave(
                nucleus_3d_file, nucleus.astype(self.raw_image.dtype), imagej=True
            )
            tifffile.imsave(
                nucleus_2d_file, nucleus_max_z.astype(self.raw_image.dtype), imagej=True
            )
        logging.debug(
            "Saved {} nuclei images to {}.".format(
                len(self.nuclear_crops), self.output_dir
            )
        )

    def compute_nuclei_masks(
        self,
        method: str = "morph_snakes",
        median_smoothing=False,
        min_size: int = 400,
        n_jobs: int = 10,
        lambda1: float = 1,
        lambda2: float = 2,
        zmin: int = 5,
        zmax: int = 20,
        **kwargs
    ):
        dna_channel_id = self.channels.index("dna")
        res = Parallel(n_jobs=n_jobs)(
            delayed(get_nuclear_mask_in_3d)(
                dna_image=self.nuclear_crops[i][:, dna_channel_id],
                method=method,
                median_smoothing=median_smoothing,
                min_size=min_size,
                lambda1=lambda1,
                lambda2=lambda2,
                zmin=zmin,
                zmax=zmax,
                **kwargs
            )
            for i in tqdm(
                range(len(self.nuclear_crops)), desc="Compute 3D nuclei masks"
            )
        )
        self.nuclei_masks = []
        self.qc_results = []
        for item in res:
            self.nuclei_masks.append(item[0])
            self.qc_results.append(item[1])

    def read_in_external_nuclear_masks(self, mask_dir: str):
        self.nuclei_masks = []
        for i in tqdm(
            range(len(self.nuclear_crops)), desc="Read in external 3D nuclei images"
        ):
            loc = os.path.join(mask_dir, self.nuclei_ids[i] + ".tif")
            mask = imread(loc)[:, :, :, -1]
            self.nuclei_masks.append(mask)

    def add_nuclei_mask_channel(self):
        self.channels.append("nuclear_mask")
        for i in tqdm(range(len(self.nuclei_masks)), desc="Add nuclear mask channel"):
            self.nuclear_crops[i] = np.concatenate(
                [
                    self.nuclear_crops[i],
                    np.expand_dims(self.nuclei_masks[i], axis=1).astype(
                        self.nuclear_crops[i].dtype
                    ),
                ],
                axis=1,
            )

    def plot_colored_nuclei_masks(self):
        dna_channel_id = self.channels.index("dna")
        output_dir = os.path.join(self.output_dir, "colored_nuclei_masks")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        for i in range(len(self.nuclear_crops)):
            dna_image = self.nuclear_crops[i][:, dna_channel_id]
            nuclear_mask = self.nuclei_masks[i]
            fig, ax = plot_colored_3d_segmentation(
                mask=nuclear_mask, intensity_image=dna_image
            )
            file_name = os.path.join(output_dir, self.nuclei_ids[i] + ".png")
            save_figure_as_png(fig=fig, file=file_name)
            plt.close()

    def save_nuclei_images(self, nuclei_output_dir: str = None):
        if nuclei_output_dir is None:
            nuclei_output_dir = "nuclei_images"

        self.nuclei_image_dir = os.path.join(self.output_dir, nuclei_output_dir)

        if not os.path.exists(self.nuclei_image_dir):
            os.makedirs(self.nuclei_image_dir, exist_ok=True)

        for i in tqdm(range(len(self.nuclear_crops)), desc="Save nuclei images"):
            nucleus = np.expand_dims(self.nuclear_crops[i], 0)
            tifffile.imsave(
                os.path.join(self.nuclei_image_dir, self.nuclei_ids[i] + ".tif"),
                nucleus,
                imagej=True,
            )

    def save_labeled_projections(self):
        self.labeled_projections_dir = os.path.join(
            self.output_dir, "labeled_segmentation_2d"
        )
        self.colored_projections_dir = os.path.join(
            self.output_dir, "colored_segmentation_2d"
        )
        if not os.path.exists(self.labeled_projections_dir):
            os.makedirs(self.labeled_projections_dir, exist_ok=True)
        if not os.path.exists(self.colored_projections_dir):
            os.makedirs(self.colored_projections_dir, exist_ok=True)
        labeled_projection_file = os.path.split(self.file_name)[1]
        labeled_projection_file_name = os.path.join(
            self.labeled_projections_dir, labeled_projection_file
        )
        colored_projection_file_name = os.path.join(
            self.colored_projections_dir,
            labeled_projection_file[: labeled_projection_file.index(".")] + ".png",
        )

        tifffile.imsave(labeled_projection_file_name, self.labeled_projection)

        scaled_z_projection = (
            (self.z_projection - self.z_projection.min())
            / (self.z_projection.max() - self.z_projection.min())
            * 255
        )
        scaled_z_projection = np.uint8(scaled_z_projection)
        colored_segmentation = color.label2rgb(
            label=self.labeled_projection, image=scaled_z_projection, bg_label=0
        )
        io.imsave(colored_projection_file_name, colored_segmentation)

    def save_raw_dna_projections(self):
        self.dna_projections_dir = os.path.join(self.output_dir, "dna_projections_2d")
        if not os.path.exists(self.dna_projections_dir):
            os.makedirs(self.dna_projections_dir, exist_ok=True)
        dna_projection_file_name = os.path.split(self.file_name)[1]
        dna_projection_file_name = os.path.join(
            self.dna_projections_dir, dna_projection_file_name
        )
        tifffile.imsave(dna_projection_file_name, self.z_projection)

    def extract_marker_labels(
        self,
        channel_id: int,
        channel_name: str,
        median_filtering: bool = True,
        min_size: int = 0,
    ):
        channel_image = self.raw_image[:, channel_id, :, :]
        channel_proj = channel_image.max(axis=0)
        if median_filtering:
            channel_proj = filters.median(channel_proj)
        tau = filters.threshold_otsu(channel_proj)
        channel_mask = channel_proj > tau
        channel_mask = ndi.binary_fill_holes(channel_mask)
        channel_mask = measure.label(channel_mask)
        channel_mask = morphology.remove_small_objects(channel_mask, min_size=min_size)
        channel_mask = channel_mask > 0
        self.labeled_marker_projections[channel_name] = (
            self.labeled_projection * channel_mask
        )

    def save_labeled_marker_projections(self, marker: str):
        self.marker_label_output_dir = os.path.join(
            self.output_dir, "{}_labeled_segmentation_2d".format(marker)
        )
        if not os.path.exists(self.marker_label_output_dir):
            os.makedirs(self.marker_label_output_dir)
        marker_labeled_projection_file = os.path.split(self.file_name)[1]
        marker_labeled_projection_file_name = os.path.join(
            self.marker_label_output_dir, marker_labeled_projection_file
        )
        tifffile.imsave(
            marker_labeled_projection_file_name, self.labeled_marker_projections[marker]
        )

    def run_segmentation_pipeline_2d(self, segmentation_2d_params_dict: dict = None):
        if "normalize_channels" in segmentation_2d_params_dict:
            channels = [
                self.channels.index(c)
                for c in segmentation_2d_params_dict["normalize_channels"]
            ]
        else:
            # channels = [self.channels.index("dna"), self.ch]
            channels = [0, 1]
        self.apply_channel_normalization(channels=channels)
        self.apply_image_filter(filter_type=segmentation_2d_params_dict["filter_type"])
        self.apply_lumination_correction(gamma=segmentation_2d_params_dict["gamma"])
        self.apply_image_filter(
            filter_type=segmentation_2d_params_dict["binary_filter_type"]
        )
        if (
            "morphological_closing" in segmentation_2d_params_dict
            and segmentation_2d_params_dict["morphological_closing"]
        ):
            self.apply_morphological_closing()
        self.clean_binary_segmentation(
            min_object_size=segmentation_2d_params_dict["min_area"]
        )
        self.segment_by_connected_components()
        self.cleaned_labeled_segmentation(
            min_area=segmentation_2d_params_dict["min_area"],
            max_area=segmentation_2d_params_dict["max_area"],
        )
        if "expansion" in segmentation_2d_params_dict:
            expansion = segmentation_2d_params_dict["expansion"]
        else:
            expansion = 1
        self.get_nuclear_crops(expansion=expansion)
        self.save_nuclear_crops()
        self.save_labeled_projections()
        self.save_raw_dna_projections()
        if "marker_channel_dict" in segmentation_2d_params_dict:
            marker_channel_dict = segmentation_2d_params_dict["marker_channel_dict"]
            marker_median_filter = segmentation_2d_params_dict["marker_median_filter"]
            marker_min_area = segmentation_2d_params_dict["marker_min_area"]
            for (
                marker_channel_id,
                marker_channel_name,
            ) in marker_channel_dict.items():
                self.extract_marker_labels(
                    channel_id=int(marker_channel_id),
                    channel_name=marker_channel_name,
                    median_filtering=marker_median_filter,
                    min_size=marker_min_area,
                )
                self.save_labeled_marker_projections(marker=marker_channel_name)

    def save_qc_results(self):
        file_path = os.path.join(self.output_dir, "qc_results.csv")
        if os.path.exists(file_path):
            self.qc_results = pd.read_csv(file_path, index_col=0).append(
                pd.DataFrame(
                    self.qc_results, index=self.nuclei_ids, columns=["qc_pass"]
                )
            )
        else:
            self.qc_results = pd.DataFrame(
                self.qc_results, index=self.nuclei_ids, columns=["qc_pass"]
            )
        self.qc_results.to_csv(file_path)

    def run_segmentation_pipeline_3d(self, segmentation_3d_params_dict: dict = None):
        if segmentation_3d_params_dict is not None:
            self.compute_nuclei_masks(**segmentation_3d_params_dict)
        else:
            self.compute_nuclei_masks()
        self.save_qc_results()
        self.add_nuclei_mask_channel()
        self.save_nuclei_images()
        self.plot_colored_nuclei_masks()

    def use_external_segmentation_3d(self, mask_dir: str, qc_file_path: str):
        qc_out_file_path = os.path.join(self.output_dir, "qc_results.csv")
        shutil.copy(qc_file_path, qc_out_file_path)
        self.read_in_external_nuclear_masks(mask_dir)
        self.add_nuclei_mask_channel()
        self.save_nuclei_images()
        self.plot_colored_nuclei_masks()

    def run_default_pipeline(
        self,
        segmentation_2d_params_dict: dict,
        segmentation_3d_params_dict: dict,
        segment_3d: bool = True,
    ):
        for i in tqdm(range(len(self.file_list)), desc="Overall segmentation process"):
            if not self.read_in_image(i):
                continue
            self.run_segmentation_pipeline_2d(segmentation_2d_params_dict)
            if segment_3d:
                self.run_segmentation_pipeline_3d(segmentation_3d_params_dict)

    def run_pipeline_with_external_3d_masks(
        self, segmentation_2d_params_dict: dict, mask_3d_dir, qc_file_path
    ):
        for i in tqdm(range(len(self.file_list)), desc="Overall segmentation process"):
            if not self.read_in_image(i):
                continue
            self.run_segmentation_pipeline_2d(segmentation_2d_params_dict)
            self.use_external_segmentation_3d(
                mask_dir=mask_3d_dir, qc_file_path=qc_file_path
            )
