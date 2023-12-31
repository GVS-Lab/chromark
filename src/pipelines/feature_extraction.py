import os
from typing import List

import numpy as np
import pandas as pd
import tifffile
from nmco.utils.run_nuclear_feature_extraction import run_nuclear_chromatin_feat_ext
from tqdm import tqdm

from src.utils.basic.feature_extraction import (
    compute_all_morphological_chromatin_features_3d,
    compute_all_channel_features,
    expand_boundaries,
    get_n_foci,
    get_2D_foci_features,
)
from src.utils.basic.io import get_file_list


class FeatureExtractionPipeline(object):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir


class DnaFeatureExtractionPipeline2D(FeatureExtractionPipeline):
    def __init__(
        self, output_dir: str, dna_projections_dir: str, labeled_projections_dir: str
    ):
        super().__init__(output_dir=output_dir)
        self.dna_projections_dir = dna_projections_dir
        self.labeled_projections_dir = labeled_projections_dir

        self.raw_nuclei_image_locs = get_file_list(self.dna_projections_dir)
        self.nuclei_mask_locs = get_file_list(self.labeled_projections_dir)
        self.features = []

    def extract_chromatin_features(self):
        all_features = []
        all_ids = []

        for i in tqdm(range(len(self.raw_nuclei_image_locs))):
            nuclei_image_loc = self.raw_nuclei_image_locs[i]
            nuclei_mask_loc = self.nuclei_mask_locs[i]
            nuclei_image_id = os.path.split(nuclei_image_loc)[1]
            nuclei_image_id = nuclei_image_id[: nuclei_image_id.index(".")]
            nuclei_features = run_nuclear_chromatin_feat_ext(
                nuclei_image_loc,
                nuclei_mask_loc,
                "temp/",
                step_size_curvature=5,
                hc_threshold=1.5,
                gclm_lengths=[5, 25, 100],
                normalize=True,
                prominance_curvature=0.1,
            )
            nuclei_ids = [
                nuclei_image_id + "_{}".format(i) for i in range(len(nuclei_features))
            ]

            all_features.append(nuclei_features)
            all_ids.extend(nuclei_ids)

        self.features = pd.concat(all_features)
        self.features.index = all_ids

    def add_marker_labels(self, labeled_marker_image_dir: str, marker: str):
        # Todo use label as final identifier not label-1 --> requires appropriate change in extract_chromatin function

        labeled_marker_image_locs = get_file_list(labeled_marker_image_dir)
        self.features[marker] = np.repeat(0, len(self.features))
        for i in range(len(labeled_marker_image_locs)):
            labeled_marker_image_loc = labeled_marker_image_locs[i]
            labeled_marker_image = tifffile.imread(labeled_marker_image_loc)
            labeled_marker_image_id = os.path.split(labeled_marker_image_loc)[1]
            labeled_marker_image_id = labeled_marker_image_id[
                : labeled_marker_image_id.index(".")
            ]
            for label in np.unique(labeled_marker_image):
                if label > 0:
                    self.features.loc[
                        labeled_marker_image_id + "_{}".format(label - 1), marker
                    ] = 1

    def save_features(self, file_name: str = None):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if file_name is None:
            file_name = "chromatin_features_2d.csv"
        self.features.to_csv(os.path.join(self.output_dir, file_name))

    def run_default_pipeline(
        self,
        marker_image_dirs: List = None,
        markers: List = None,
        save_features: bool = True,
    ):
        self.extract_chromatin_features()
        if marker_image_dirs is not None:
            for i in range(len(marker_image_dirs)):
                self.add_marker_labels(
                    labeled_marker_image_dir=marker_image_dirs[i], marker=markers[i]
                )
        if save_features:
            self.save_features()


class DnaFeatureExtractionPipeline3D(FeatureExtractionPipeline):
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        channels: List[str],
        segmentation_param_dict: dict = None,
    ):
        super().__init__(output_dir=output_dir)
        self.image_locs = get_file_list(input_dir)
        self.channels = [channel.lower() for channel in channels]
        self.image_ids = None
        self.raw_images = None
        self.masks = None
        self.dna_features = None
        self.segmentation_param_dict = segmentation_param_dict

    def read_in_images(self):
        raw_images = []
        masks = []
        image_ids = []
        for i in tqdm(range(len(self.image_locs))):
            image_loc = self.image_locs[i]
            image_id = os.path.split(image_loc)[1]
            image_id = image_id[: image_id.index(".")]

            raw_image = tifffile.imread(image_loc)
            # Todo make that less error prone with searching over the channels
            raw_images.append(raw_image[:, :-1])
            masks.append(raw_image[:, -1])
            image_ids.append(image_id)
        self.raw_images = raw_images
        self.image_ids = image_ids
        self.masks = masks

    def extract_dna_features(
        self, bins: int = 10, selem: np.ndarray = None, compute_rdp: bool = True
    ):
        all_features = []
        dna_channel_id = self.channels.index("dna")
        for i in tqdm(range(len(self.raw_images)), desc="Extract DNA dna_features"):
            dna_image = self.raw_images[i][:, dna_channel_id]
            nucleus_mask = self.masks[i]
            features = compute_all_morphological_chromatin_features_3d(
                dna_image,
                nucleus_mask=nucleus_mask,
                bins=bins,
                selem=selem,
                compute_rdp=compute_rdp,
            )
            features = pd.DataFrame(features, index=[self.image_ids[i]])
            all_features.append(features)

        self.dna_features = pd.concat(all_features)
        self.dna_features.index = self.image_ids

    def save_features(self, file_name: str = None):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if file_name is None:
            file_name = "chromatin_features_3d.csv"
        self.dna_features.to_csv(os.path.join(self.output_dir, file_name))

    def run_default_pipeline(self, segmentation_params_dict: dict = None):
        self.read_in_images()
        self.extract_dna_features()
        self.save_features()


class MultiChannelFeatureExtractionPipeline3D(DnaFeatureExtractionPipeline3D):
    def __init__(self, input_dir: str, output_dir: str, channels: List[str]):
        super().__init__(input_dir=input_dir, output_dir=output_dir, channels=channels)
        self.features = None

    def read_in_images(self):
        super().read_in_images()

    def extract_dna_features(
        self, bins: int = 10, selem: np.ndarray = None, compute_rdp: bool = True
    ):
        super().extract_dna_features(bins=bins, selem=selem, compute_rdp=compute_rdp)
        if self.features is None:
            self.features = self.dna_features
        else:
            self.features = pd.concat(([self.features, self.dna_features]), axis=1)

    def extract_foci_characteristics(
        self, channel, alpha=2.5, min_dist=1, min_size=4, sigma=0.5, expansion=0,
    ):
        channel_id = self.channels.index(channel)
        all_foci_feats = []
        for i in tqdm(
            range(len(self.raw_images)),
            desc="Extract {} foci counts".format(channel.upper()),
        ):
            channel_image = self.raw_images[i][:, channel_id]
            nucleus_mask = expand_boundaries(self.masks[i], expansion)
            image_id = self.image_ids[i]
            foci_feats = get_2D_foci_features(
                image=channel_image,
                mask=nucleus_mask,
                image_index=image_id,
                alpha=alpha,
                min_dist=min_dist,
                min_size=min_size,
            )
            all_foci_feats.append(foci_feats)
        all_foci_feats = pd.concat(all_foci_feats)
        all_foci_feats.index = list(range(len(all_foci_feats)))
        return all_foci_feats

    def extract_foci_counts(
        self,
        channel: str,
        z_project=False,
        min_dist=3,
        threshold_rel: float = None,
        expansion=0,
    ):
        channel_id = self.channels.index(channel)
        all_channel_foci = []

        for i in tqdm(
            range(len(self.raw_images)),
            desc="Extract {} foci counts".format(channel.upper()),
        ):
            channel_image = self.raw_images[i][:, channel_id]
            nucleus_mask = expand_boundaries(self.masks[i], expansion)
            features = get_n_foci(
                channel_image,
                mask=nucleus_mask,
                description=channel,
                index=i,
                z_project=z_project,
                min_dist=min_dist,
                threshold_rel=threshold_rel,
            )
            all_channel_foci.append(features)

        channel_features = pd.concat(all_channel_foci)
        channel_features.index = self.image_ids
        if self.features is None:
            self.features = channel_features
        else:
            self.features = pd.concat([self.features, channel_features], axis=1)

    def extract_channel_features(
        self, channel: str, expansion: int = 0, z_project: bool = False
    ):
        channel_id = self.channels.index(channel)
        all_channel_features = []
        for i in tqdm(
            range(len(self.raw_images)),
            desc="Extract {} features".format(channel.upper()),
        ):
            channel_image = self.raw_images[i][:, channel_id]
            nucleus_mask = expand_boundaries(self.masks[i], expansion)
            features = compute_all_channel_features(
                channel_image,
                nucleus_mask=nucleus_mask,
                channel=channel,
                index=i,
                z_project=z_project,
            )
            all_channel_features.append(features)
        channel_features = pd.concat(all_channel_features)
        channel_features.index = self.image_ids
        if self.features is None:
            self.features = channel_features
        else:
            self.features = pd.concat([self.features, channel_features], axis=1)

    def save_features(self, features=None, file_name: str = None):
        if features is None:
            features = self.features
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if file_name is None:
            file_name = "nuclei_features_3d.csv"
        features.to_csv(os.path.join(self.output_dir, file_name))

    def run_default_pipeline(
        self,
        characterize_channels: List = None,
        compute_rdp: bool = True,
        save_features: bool = True,
        protein_expansions: List[int] = None,
    ):
        if protein_expansions is None:
            protein_expansions = [0] * len(characterize_channels)
        self.read_in_images()
        self.extract_dna_features(compute_rdp=compute_rdp)
        self.extract_channel_features(channel="dna")
        if characterize_channels is None:
            characterize_channels = []
        for i in range(len(characterize_channels)):
            channel = characterize_channels[i]
            self.extract_channel_features(
                channel=channel, expansion=protein_expansions[i], z_project=True
            )
            self.extract_channel_features(
                channel=channel, expansion=protein_expansions[i], z_project=False
            )
            if channel.lower() == "gh2ax":
                # self.extract_foci_counts(
                #     channel=channel, min_dist=3, threshold_rel=1, expansion=0
                # )
                foci_characteristics = self.extract_foci_characteristics(
                    channel=channel, min_dist=1, min_size=4, alpha=2.5, sigma=0.5,
                )
                self.save_features(
                    features=foci_characteristics,
                    file_name="{}_foci_feats.csv".format(channel),
                )

        if save_features:
            self.save_features(features=self.features)

    def run_gh2ax_foci_pipeline(
        self, characterize_channels: List = None, protein_expansions: List[int] = None,
    ):
        if protein_expansions is None:
            protein_expansions = [0] * len(characterize_channels)
        self.read_in_images()
        if characterize_channels is None:
            characterize_channels = []
        for i in range(len(characterize_channels)):
            channel = characterize_channels[i]
            if channel.lower() == "gh2ax":
                foci_characteristics = self.extract_foci_characteristics(
                    channel=channel, min_dist=1, min_size=4, alpha=2.5, sigma=0.5,
                )
                self.save_features(
                    features=foci_characteristics,
                    file_name="{}_foci_feats.csv".format(channel),
                )
