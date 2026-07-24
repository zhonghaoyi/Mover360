import torch
import os
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import numpy as np
import random
import math
from collections import defaultdict
from utils.pano import Equirectangular, random_sample_camera, horizon_sample_camera, icosahedron_sample_camera, cubemap_sample_camera, eight_pers_sample_camera
import lightning as L
import cv2
from glob import glob
from einops import rearrange
from abc import abstractmethod
from PIL import Image
from external.Perspective_and_Equirectangular import mp2e
import json


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_DINO_REF_JSON_PATH = os.path.join(
    REPO_ROOT,
    'Dataprocess',
    'dino_filter_outputs',
    'panorama_top3_references.json',
)


def get_K_R(FOV, THETA, PHI, height, width):
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    return K, R


class PanoDataset(torch.utils.data.Dataset):
    _external_ref_index_cache = {}
    _dino_ref_lookup_cache = {}

    def __init__(self, config, mode='train'):
        self.mode = mode
        self.random_clip = 0
        self.data_dir = config['data_dir']
        self.inpaint_data_dir = config['inpaint_data_dir']
        self.predict_data_dir = config['predict_data_dir']
        self.s3d_data_dir = config['s3d_data_dir']
        self.s3d_inpaint_data_dir = config['s3d_inpaint_data_dir']
        self.result_dir = config.get('result_dir', None)
        self.config = config
        self.use_txt_prompt = bool(config.get('use_txt_prompt', True))
        self.test_function = str(config.get('test_function', 'move')).lower()
        self.use_fixed_pers_prompt = config['use_fixed_pers_prompt']
        self.use_cubemap_prompt = config['use_cubemap_prompt']
        self.only_pano = config['only_pano']
        self.use_ref = config['use_ref']
        self.guidance_full_mask_prob = min(
            max(float(config.get('guidance_full_mask_prob', 0.5)), 0.0),
            1.0,
        )
        self.external_ref_base_dir = config.get(
            'external_ref_base_dir',
            'data/UE5_data/Scene_pers1',
        )
        self.external_ref_saved_dirs = ('Saved_1', 'Saved_2')
        self.external_ref_cameras = tuple(f'CineCameraActor_{i}' for i in range(1, 5))
        self.dino_ref_json_path = config.get(
            'dino_ref_json_path',
            DEFAULT_DINO_REF_JSON_PATH,
        )
        self.dino_ref_selection = str(config.get('dino_ref_selection', 'first')).lower()
        self.dino_ref_fallback_to_external = bool(config.get('dino_ref_fallback_to_external', True))
        self.dino_ref_strict = bool(config.get('dino_ref_strict', False))

        self.data = self.load_split(mode)

        if mode == 'predict':
            self.data = sum([[d.copy() for i in range(self.config['repeat_predict'])] for d in self.data], [])
            if self.config['repeat_predict'] > 1:
                for i, d in enumerate(self.data):
                    d['repeat_id'] = i % self.config['repeat_predict']

        if not self.config['gt_as_result'] and self.result_dir is not None:
            results = self.scan_results(self.result_dir)
            assert results, f"No results found in {self.result_dir}, forgot to set environment variable WANDB_RUN_ID?"
            
            results_set = set(results)
            new_data = [d for d in self.data if (d['scene_id'], d['view_id']) in results_set]
            if len(new_data) != len(self.data):
                print(f"WARNING: {len(self.data)-len(new_data)} views are missing in results folder {self.result_dir} for {self.mode} set.")
                self.data = list(new_data)
                self.data.sort()

    @abstractmethod  #sub class must implement this method
    def load_split(self):
        pass

    @abstractmethod
    def scan_results(self):
        pass

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _normalize_source_mask_type(mask_type):
        normalized = str(mask_type or 'bbox').strip().lower()
        if normalized in {'bbox', 'box'}:
            return 'bbox'
        if normalized in {'full', 'mask', 'fine'}:
            return 'full'
        raise ValueError(
            "inference_source_mask_type must be one of {'bbox', 'full'}, "
            f"got {mask_type!r}"
        )

    def _select_guidance_mask_path(self, box_mask_path, full_mask_path):
        if full_mask_path is None:
            return box_mask_path
        if box_mask_path is None:
            return full_mask_path
        # Mix bbox and fine masks only during training; each mask channel samples independently.
        if self.mode != 'train':
            return box_mask_path
        if random.random() < self.guidance_full_mask_prob:
            return full_mask_path
        return box_mask_path

    def _select_source_guidance_mask_path(self, box_mask_path, full_mask_path):
        if self.mode == 'train':
            return self._select_guidance_mask_path(box_mask_path, full_mask_path)

        source_mask_type = self._normalize_source_mask_type(
            self.config.get('inference_source_mask_type', 'bbox')
        )
        if source_mask_type == 'full':
            return full_mask_path or box_mask_path
        return box_mask_path or full_mask_path

    def _resolve_point_mask_path(self, mask_path):
        if mask_path is None:
            return None

        for mask_root in ("MovieRenders_ObjectMaskPoint", "MovieRenders_ObjectMaskBox", "MovieRenders_ObjectMask"):
            source_token = f"{os.sep}{mask_root}{os.sep}"
            if source_token not in mask_path:
                continue
            point_path = mask_path.replace(
                source_token,
                f"{os.sep}MovieRenders_ObjectMaskPoint{os.sep}",
                1,
            )
            if os.path.exists(point_path):
                return point_path
            if not getattr(self, "_warned_missing_point_mask", False):
                print(
                    "Warning: Cannot find precomputed point mask file under "
                    f"MovieRenders_ObjectMaskPoint: {point_path}. "
                    "Using a zero point channel for missing point maps."
                )
                self._warned_missing_point_mask = True
            return None

        return None

    @staticmethod
    def _resolve_input_condition_rgb_path(data):
        function = str(data.get('function', '')).lower()
        if function == 'remove':
            return data.get('pano_path') or data.get('remove_pano_path')
        return data.get('remove_pano_path') or data.get('pano_path')

    @staticmethod
    def _resolve_depth_path_from_rgb_path(rgb_path):
        if rgb_path is None:
            return None

        source_token = f"{os.sep}MovieRenders_Normal{os.sep}"
        target_token = f"{os.sep}MovieRenders_PredictDepth{os.sep}"
        if source_token not in rgb_path:
            return None

        depth_path = rgb_path.replace(source_token, target_token, 1)
        depth_path = os.path.splitext(depth_path)[0] + '.npy'
        return depth_path

    @staticmethod
    def _resolve_gt_depth_path_from_rgb_path(
        rgb_path,
        depth_root_name='MovieRenders_SceneDepth_1024_gt',
    ):
        if rgb_path is None:
            return None

        source_token = f"{os.sep}MovieRenders_Normal{os.sep}"
        target_token = f"{os.sep}{depth_root_name}{os.sep}"
        if source_token not in rgb_path:
            return None

        depth_path = rgb_path.replace(source_token, target_token, 1)
        depth_dir = os.path.dirname(depth_path)
        stem = os.path.splitext(os.path.basename(depth_path))[0]

        candidates = []
        prefix_and_frame = stem.rsplit('_', 1)
        if len(prefix_and_frame) == 2 and prefix_and_frame[1].isdigit():
            camera_prefix, frame_id = prefix_and_frame
            candidates.append(os.path.join(depth_dir, f"{camera_prefix}_.{frame_id}.exr"))
        candidates.append(os.path.join(depth_dir, f"{stem}.exr"))

        expanded_candidates = []
        for candidate in candidates:
            expanded_candidates.append(candidate)
            expanded_candidates.append(os.path.splitext(candidate)[0] + '.EXR')

        for candidate in expanded_candidates:
            if os.path.exists(candidate):
                return candidate
        return expanded_candidates[0] if expanded_candidates else None

    @staticmethod
    def _read_exr_depth(depth_path):
        image = cv2.imread(
            depth_path,
            cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
        )
        if image is None:
            try:
                import OpenEXR
                import Imath
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot read EXR depth with OpenCV and OpenEXR is unavailable: {depth_path}"
                ) from exc

            exr_file = OpenEXR.InputFile(depth_path)
            header = exr_file.header()
            data_window = header['dataWindow']
            width = data_window.max.x - data_window.min.x + 1
            height = data_window.max.y - data_window.min.y + 1
            channels = header.get('channels', {})
            channel_name = next(
                (name for name in ('R', 'Y', 'Z', 'A') if name in channels),
                next(iter(channels), None),
            )
            if channel_name is None:
                raise RuntimeError(f"EXR depth file has no channels: {depth_path}")
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
            channel_bytes = exr_file.channel(channel_name, pixel_type)
            image = np.frombuffer(channel_bytes, dtype=np.float32).reshape(height, width)

        depth = np.asarray(image, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = np.squeeze(depth)
        if depth.ndim != 2:
            raise ValueError(f"Expected a 2D EXR depth map, got shape {depth.shape}")
        depth = np.where(np.isfinite(depth), depth, 0.0)
        depth = np.maximum(depth, 0.0)
        return depth.astype(np.float32)

    @staticmethod
    def _normalize_depth_for_vae(
        depth_map,
        lower_quantile=0.02,
        upper_quantile=0.98,
        use_log=False,
    ):
        depth = np.asarray(depth_map, dtype=np.float32).squeeze()
        if depth.ndim != 2:
            raise ValueError(f"Expected a 2D depth map, got shape {depth.shape}")

        valid = np.isfinite(depth) & (depth > 0)
        if not np.any(valid):
            return np.zeros_like(depth, dtype=np.float32)

        depth = depth.copy()
        if use_log:
            depth[valid] = np.log1p(depth[valid])

        lower_quantile = float(lower_quantile)
        upper_quantile = float(upper_quantile)
        lower_quantile = min(max(lower_quantile, 0.0), 1.0)
        upper_quantile = min(max(upper_quantile, 0.0), 1.0)
        if upper_quantile < lower_quantile:
            lower_quantile, upper_quantile = upper_quantile, lower_quantile

        valid_depth = depth[valid]
        lo = float(np.quantile(valid_depth, lower_quantile))
        hi = float(np.quantile(valid_depth, upper_quantile))

        normalized = np.zeros_like(depth, dtype=np.float32)
        if hi <= lo + 1e-6:
            normalized[valid] = 0.5
            return normalized

        normalized[valid] = (depth[valid] - lo) / (hi - lo)
        normalized = np.clip(normalized, 0.0, 1.0)
        return normalized.astype(np.float32)

    def _build_mover360_batch_output(self, data):
        # Keep the collated batch schema explicit for Mover360_Base so task-specific
        # assembly fields never leak into default_collate.
        keep_keys = (
            'scene_id',
            'camera_id',
            'object_id',
            'function',
            'pano_id',
            'instruction_text',
            'prompt',
            'cameras',
            'height',
            'width',
            'pano',
            'images',
            'refs',
            'input_depth',
            'target_depth',
            'target_depth_path',
            'remove_pano',
            'remove_images',
            'pano_mask',
            'pano_full_mask',
            'inference_source_mask_type',
            'pano_prompt',
            'pano_prompt_without_img',
            'pano_prompt_with_mask',
            'pano_prompt_with_ref',
            'pano_prompt_with_ref_and_mask',
            'has_ref',
            'ref_img_path',
            'ref_source_id',
            'pano_pred',
            'images_pred',
            'test_pers',
            'ori_test_pers',
        )

        batch = {}
        for key in keep_keys:
            value = data.get(key)
            if value is not None:
                batch[key] = value

        batch.setdefault('has_ref', False)
        batch.setdefault('ref_img_path', '')
        batch.setdefault('ref_source_id', '')
        return batch

    @staticmethod
    def normalize_prompt_text(prompt):
        prompt = prompt.strip()
        if prompt.endswith('.'):
            prompt = prompt[:-1]
        return prompt

    @staticmethod
    def ensure_prompt_sentence(prompt):
        prompt = prompt.strip()
        if prompt and prompt[-1] not in '.!?':
            prompt = prompt + '.'
        return prompt

    def read_prompt_text(self, path):
        with open(path) as f:
            prompt = f.readlines()[0]
        return self.normalize_prompt_text(prompt)

    @staticmethod
    def predefined_prompt_text(function):
        function = str(function).lower()
        if function == 'add':
            return 'add the object'
        if function == 'remove':
            return 'remove the object'
        return 'move the object'

    def read_or_predefined_prompt_text(self, path, function):
        if self.use_txt_prompt:
            return self.read_prompt_text(path)
        return self.predefined_prompt_text(function)

    def build_flux2_text_only_prompt(self, prompt):
        return self.ensure_prompt_sentence(self.normalize_prompt_text(prompt))

    def build_flux2_input_prompt(self, prompt):
        prompt = self.build_flux2_text_only_prompt(prompt)
        return f"Image 1 is the panorama to edit. {prompt}"

    def build_flux2_add_text_only_prompt(self, prompt):
        base_prompt = self.build_flux2_text_only_prompt(prompt)
        normalized_prompt = self.normalize_prompt_text(prompt).lower()
        explicit_add_prompt = "Add the object at the target region."
        if normalized_prompt in {"", "move the object", "move object", "add the object", "add object"}:
            return explicit_add_prompt
        return f"{explicit_add_prompt} {base_prompt}"

    def build_flux2_add_input_prompt(self, prompt):
        prompt = self.build_flux2_add_text_only_prompt(prompt)
        return f"Image 1 is the panorama to edit. {prompt}"

    def build_flux2_add_mask_prompt(self, prompt):
        prompt = self.build_flux2_add_text_only_prompt(prompt)
        return (
            "Image 1 is the panorama to edit. "
            "Image 2 marks the target region. "
            f"{prompt}"
        )

    def build_flux2_ref_prompt(self, prompt):
        prompt = self.build_flux2_add_text_only_prompt(prompt)
        return (
            "Image 1 is the panorama to edit. "
            "Image 2 is the object reference. "
            f"{prompt}"
        )

    def build_flux2_add_ref_and_mask_prompt(self, prompt):
        prompt = self.build_flux2_add_text_only_prompt(prompt)
        return (
            "Image 1 is the panorama to edit. "
            "Image 2 is the object reference. "
            "Image 3 marks the target region. "
            f"{prompt}"
        )

    def build_flux2_move_text_only_prompt(self, prompt):
        base_prompt = self.build_flux2_text_only_prompt(prompt)
        normalized_prompt = self.normalize_prompt_text(prompt).lower()
        if 'source' in normalized_prompt and 'target' in normalized_prompt:
            return base_prompt

        explicit_move_prompt = "Move the object from the source region to the target region."
        if normalized_prompt in {"", "move the object", "move object"}:
            return explicit_move_prompt
        return f"{explicit_move_prompt} {base_prompt}"

    def build_flux2_move_input_prompt(self, prompt):
        prompt = self.build_flux2_move_text_only_prompt(prompt)
        return f"Image 1 is the panorama to edit. {prompt}"

    def build_flux2_move_mask_prompt(self, prompt):
        prompt = self.build_flux2_move_text_only_prompt(prompt)
        return (
            "Image 1 is the panorama to edit. "
            "Image 2 marks the source and target regions. "
            f"{prompt}"
        )

    def build_flux2_move_ref_prompt(self, prompt):
        prompt = self.build_flux2_move_text_only_prompt(prompt)
        return (
            "Image 1 is the panorama to edit. "
            "Image 2 is the object reference. "
            f"{prompt}"
        )

    def build_flux2_move_ref_and_mask_prompt(self, prompt):
        prompt = self.build_flux2_move_text_only_prompt(prompt)
        return (
            "Image 1 is the panorama to edit. "
            "Image 2 is the object reference. "
            "Image 3 marks the source and target regions. "
            f"{prompt}"
        )

    def build_flux2_remove_text_only_prompt(self, prompt):
        base_prompt = self.build_flux2_text_only_prompt(prompt)
        normalized_prompt = self.normalize_prompt_text(prompt).lower()
        explicit_remove_prompt = "Remove the object from the source region."
        if normalized_prompt in {"", "move the object", "move object", "remove the object", "remove object"}:
            return explicit_remove_prompt
        return f"{explicit_remove_prompt} {base_prompt}"

    def build_flux2_remove_input_prompt(self, prompt):
        prompt = self.build_flux2_remove_text_only_prompt(prompt)
        return f"Image 1 is the panorama to edit. {prompt}"

    def build_flux2_remove_mask_prompt(self, prompt):
        prompt = self.build_flux2_remove_text_only_prompt(prompt)
        return (
            "Image 1 is the panorama to edit. "
            "Image 2 marks the object to remove. "
            f"{prompt}"
        )

    def build_flux2_remove_ref_prompt(self, prompt):
        prompt = self.build_flux2_remove_text_only_prompt(prompt)
        return (
            "Image 1 is the panorama to edit. "
            "Image 2 is the object reference. "
            f"{prompt}"
        )

    def build_flux2_remove_ref_and_mask_prompt(self, prompt):
        prompt = self.build_flux2_remove_text_only_prompt(prompt)
        return (
            "Image 1 is the panorama to edit. "
            "Image 2 is the object reference. "
            "Image 3 marks the object to remove. "
            f"{prompt}"
        )

    def load_prompt_without_img_add(self, path):
        prompt = self.read_or_predefined_prompt_text(path, 'add')
        return self.build_flux2_add_text_only_prompt(prompt)

    def load_prompt_without_img_move(self, path):
        prompt = self.read_or_predefined_prompt_text(path, 'move')
        return self.build_flux2_move_text_only_prompt(prompt)

    def load_prompt_without_img_remove(self, path):
        prompt = self.read_or_predefined_prompt_text(path, 'remove')
        return self.build_flux2_remove_text_only_prompt(prompt)

    def load_prompt_add(self, path):
        return self.build_flux2_add_input_prompt(self.read_or_predefined_prompt_text(path, 'add'))

    def load_prompt_add_with_mask(self, path):
        return self.build_flux2_add_mask_prompt(self.read_or_predefined_prompt_text(path, 'add'))

    def load_prompt_move(self, path):
        return self.build_flux2_move_input_prompt(self.read_or_predefined_prompt_text(path, 'move'))

    def load_prompt_with_ref(self, path):
        return self.build_flux2_ref_prompt(self.read_or_predefined_prompt_text(path, 'add'))

    def load_prompt_with_ref_and_mask(self, path):
        return self.build_flux2_add_ref_and_mask_prompt(self.read_or_predefined_prompt_text(path, 'add'))

    def load_prompt_move_with_mask(self, path):
        return self.build_flux2_move_mask_prompt(self.read_or_predefined_prompt_text(path, 'move'))

    def load_prompt_move_with_ref(self, path):
        return self.build_flux2_move_ref_prompt(self.read_or_predefined_prompt_text(path, 'move'))

    def load_prompt_move_with_ref_and_mask(self, path):
        return self.build_flux2_move_ref_and_mask_prompt(self.read_or_predefined_prompt_text(path, 'move'))

    def load_prompt_remove(self, path):
        return self.build_flux2_remove_input_prompt(self.read_or_predefined_prompt_text(path, 'remove'))

    def load_prompt_remove_with_mask(self, path):
        return self.build_flux2_remove_mask_prompt(self.read_or_predefined_prompt_text(path, 'remove'))

    def load_prompt_remove_with_ref(self, path):
        return self.build_flux2_remove_ref_prompt(self.read_or_predefined_prompt_text(path, 'remove'))

    def load_prompt_remove_with_ref_and_mask(self, path):
        return self.build_flux2_remove_ref_and_mask_prompt(self.read_or_predefined_prompt_text(path, 'remove'))

    def load_test_pers_prompt(self, path):
        with open(path) as f:
            prompt = f.readlines()[0]
        prompt = prompt.strip()
        return prompt

    @staticmethod
    def _list_image_files(folder_path):
        if folder_path is None or not os.path.isdir(folder_path):
            return []
        image_exts = ('.png', '.jpg', '.jpeg')
        return sorted(
            os.path.join(folder_path, name)
            for name in os.listdir(folder_path)
            if name.lower().endswith(image_exts)
        )

    @staticmethod
    def _extract_object_prefix(object_id):
        if not object_id:
            return None
        normalized = object_id[:-3] if object_id.endswith('_bj') else object_id
        return normalized.rsplit('_seq', 1)[0]

    @classmethod
    def _build_external_ref_index(cls, base_dir):
        cached_index = cls._external_ref_index_cache.get(base_dir)
        if cached_index is not None:
            return cached_index

        index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        saved_dirs = ('Saved_1', 'Saved_2')
        camera_names = tuple(f'CineCameraActor_{i}' for i in range(1, 5))

        for saved_dir in saved_dirs:
            normal_root = os.path.join(base_dir, saved_dir, 'MovieRenders_Normal')
            mask_root = os.path.join(base_dir, saved_dir, 'MovieRenders_ObjectMask')
            if not os.path.isdir(normal_root):
                continue

            for camera_name in camera_names:
                normal_camera_dir = os.path.join(normal_root, camera_name)
                mask_camera_dir = os.path.join(mask_root, camera_name)
                if not os.path.isdir(normal_camera_dir):
                    continue

                for object_name in sorted(os.listdir(normal_camera_dir)):
                    normal_object_dir = os.path.join(normal_camera_dir, object_name)
                    if not os.path.isdir(normal_object_dir):
                        continue

                    mask_object_dir = os.path.join(mask_camera_dir, object_name)
                    image_files = cls._list_image_files(normal_object_dir)
                    mask_files = cls._list_image_files(mask_object_dir)
                    if not image_files or not mask_files:
                        continue

                    object_prefix = cls._extract_object_prefix(object_name)
                    if object_prefix is None:
                        continue

                    index[object_prefix][saved_dir][camera_name].append(
                        {
                            'saved_dir': saved_dir,
                            'camera_name': camera_name,
                            'object_name': object_name,
                            'image_path': image_files[0],
                            'mask_path': mask_files[0],
                        }
                    )

        cls._external_ref_index_cache[base_dir] = index
        return index

    @staticmethod
    def _normalize_lookup_path(path):
        if not path:
            return None
        return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))

    @staticmethod
    def _derive_mask_path_from_ref_image_path(image_path):
        source_token = f"{os.sep}MovieRenders_Normal{os.sep}"
        target_token = f"{os.sep}MovieRenders_ObjectMask{os.sep}"
        if image_path is None or source_token not in image_path:
            return None
        return image_path.replace(source_token, target_token, 1)

    @staticmethod
    def _source_id_from_reference_path(image_path):
        if not image_path:
            return None

        parts = os.path.normpath(image_path).split(os.sep)
        try:
            normal_idx = parts.index('MovieRenders_Normal')
        except ValueError:
            return None

        if normal_idx < 1 or len(parts) <= normal_idx + 3:
            return None
        saved_dir = parts[normal_idx - 1]
        camera_name = parts[normal_idx + 1]
        object_name = parts[normal_idx + 2]
        image_name = parts[-1]
        return f"{saved_dir}/{camera_name}/{object_name}/{image_name}"

    @staticmethod
    def _source_id_from_panorama_path(pano_image_path):
        if not pano_image_path:
            return None

        parts = os.path.normpath(pano_image_path).split(os.sep)
        try:
            normal_idx = parts.index('MovieRenders_Normal')
        except ValueError:
            return None

        if normal_idx < 2 or len(parts) <= normal_idx + 3:
            return None
        scene_name = parts[normal_idx - 2]
        camera_name = parts[normal_idx + 1]
        object_name = parts[normal_idx + 2]
        image_name = parts[-1]
        return f"{scene_name}/{camera_name}/{object_name}/{image_name}"

    @classmethod
    def _dino_ref_entry_from_json(cls, ref_data):
        if isinstance(ref_data, str):
            image_path = ref_data
            ref_entry = {}
        elif isinstance(ref_data, dict):
            image_path = ref_data.get('image_path')
            ref_entry = dict(ref_data)
        else:
            return None

        if not image_path:
            return None

        mask_path = ref_entry.get('mask_path') or cls._derive_mask_path_from_ref_image_path(image_path)
        if not mask_path:
            return None

        source_id = ref_entry.get('source_id') or cls._source_id_from_reference_path(image_path)
        return {
            'image_path': image_path,
            'mask_path': mask_path,
            'source_id': source_id,
            'object_name': ref_entry.get('object_name'),
            'rank': ref_entry.get('rank'),
            'score': ref_entry.get('score'),
        }

    @classmethod
    def _dedupe_dino_ref_entries(cls, entries):
        deduped = []
        seen = set()
        for entry in entries:
            image_key = cls._normalize_lookup_path(entry.get('image_path'))
            mask_key = cls._normalize_lookup_path(entry.get('mask_path'))
            key = (image_key, mask_key)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)
        return deduped

    @classmethod
    def _add_dino_lookup_entries(cls, mapping, key, entries):
        if not key or not entries:
            return
        mapping.setdefault(key, []).extend(entries)

    @classmethod
    def _build_dino_ref_lookup(cls, json_path):
        lookup = {
            'by_image_path': {},
            'by_source_id': {},
            'error': None,
        }
        if not json_path:
            return lookup

        json_path = cls._normalize_lookup_path(json_path)
        cached_lookup = cls._dino_ref_lookup_cache.get(json_path)
        if cached_lookup is not None:
            return cached_lookup

        if not os.path.exists(json_path):
            lookup['error'] = f"DINO reference JSON not found: {json_path}"
            cls._dino_ref_lookup_cache[json_path] = lookup
            return lookup

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except Exception as exc:
            lookup['error'] = f"Cannot read DINO reference JSON {json_path}: {exc}"
            cls._dino_ref_lookup_cache[json_path] = lookup
            return lookup

        ref_entries_by_source_id = {}
        for result in payload.get('results', []):
            top_refs = result.get('top_references')
            if top_refs is None:
                top_ref_paths = result.get('top_reference_image_paths', [])
                top_refs = [{'image_path': path} for path in top_ref_paths]

            ref_entries = []
            for ref_data in top_refs:
                ref_entry = cls._dino_ref_entry_from_json(ref_data)
                if ref_entry is None:
                    continue
                ref_entries.append(ref_entry)
                if ref_entry.get('source_id'):
                    ref_entries_by_source_id[ref_entry['source_id']] = ref_entry

            ref_entries = cls._dedupe_dino_ref_entries(ref_entries)
            pano_path = result.get('panorama_image_path') or result.get('image_path')
            pano_source_id = result.get('panorama_source_id') or result.get('source_id')
            image_key = cls._normalize_lookup_path(pano_path)
            cls._add_dino_lookup_entries(lookup['by_image_path'], image_key, ref_entries)
            cls._add_dino_lookup_entries(lookup['by_source_id'], pano_source_id, ref_entries)

        for pano_path, top_ref_paths in payload.get('lookup_by_panorama_image_path', {}).items():
            ref_entries = [
                cls._dino_ref_entry_from_json(ref_path)
                for ref_path in top_ref_paths
            ]
            ref_entries = cls._dedupe_dino_ref_entries(
                [entry for entry in ref_entries if entry is not None]
            )
            image_key = cls._normalize_lookup_path(pano_path)
            cls._add_dino_lookup_entries(lookup['by_image_path'], image_key, ref_entries)

        for pano_source_id, top_ref_source_ids in payload.get('lookup_by_panorama_source_id', {}).items():
            ref_entries = [
                ref_entries_by_source_id[ref_source_id]
                for ref_source_id in top_ref_source_ids
                if ref_source_id in ref_entries_by_source_id
            ]
            ref_entries = cls._dedupe_dino_ref_entries(ref_entries)
            cls._add_dino_lookup_entries(lookup['by_source_id'], pano_source_id, ref_entries)

        for mapping in (lookup['by_image_path'], lookup['by_source_id']):
            for key, entries in list(mapping.items()):
                mapping[key] = cls._dedupe_dino_ref_entries(entries)

        cls._dino_ref_lookup_cache[json_path] = lookup
        return lookup

    def _warn_once(self, attr_name, message):
        if getattr(self, attr_name, False):
            return
        print(message)
        setattr(self, attr_name, True)

    def _lookup_dino_reference_entries(self, pano_image_path):
        lookup = self._build_dino_ref_lookup(self.dino_ref_json_path)
        if lookup.get('error'):
            if self.dino_ref_strict:
                raise FileNotFoundError(lookup['error'])
            self._warn_once(
                '_warned_missing_dino_ref_json',
                f"Warning: {lookup['error']}. Reference images from DINO JSON will be disabled.",
            )
            return []

        image_key = self._normalize_lookup_path(pano_image_path)
        entries = lookup['by_image_path'].get(image_key, []) if image_key is not None else []
        if not entries:
            pano_source_id = self._source_id_from_panorama_path(pano_image_path)
            entries = lookup['by_source_id'].get(pano_source_id, []) if pano_source_id else []

        if not entries:
            if self.dino_ref_fallback_to_external:
                self._warn_once(
                    '_warned_missing_dino_ref_match',
                    "Warning: no DINO reference JSON match for panorama image; "
                    f"falling back to Scene_pers1 references: {pano_image_path}",
                )
            else:
                self._warn_once(
                    '_warned_missing_dino_ref_match',
                    f"Warning: no DINO reference JSON match for panorama image: {pano_image_path}",
                )
            return []

        entries = list(entries)
        if self.dino_ref_selection == 'random':
            random.shuffle(entries)
        elif self.dino_ref_selection not in {'first', 'top1', 'best'}:
            self._warn_once(
                '_warned_invalid_dino_ref_selection',
                f"Warning: unknown dino_ref_selection={self.dino_ref_selection!r}; using the first JSON reference.",
            )
        return entries

    def _load_dino_reference_image(self, pano_image_path, mode='train'):
        ref_entries = self._lookup_dino_reference_entries(pano_image_path)
        for ref_entry in ref_entries:
            ref_crop, loaded_ref_entry = self._load_reference_image_from_paths(
                ref_entry['image_path'],
                ref_entry['mask_path'],
                mode=mode,
                wrap_horizontal=False,
                ref_entry=ref_entry,
            )
            if ref_crop is not None:
                return ref_crop, loaded_ref_entry

        if ref_entries:
            self._warn_once(
                '_warned_unreadable_dino_ref',
                f"Warning: DINO reference entries exist but none could be loaded for {pano_image_path}",
            )
        return None, None

    @staticmethod
    def _mask_bbox_xyxy(mask_img):
        ys, xs = np.where(mask_img > 0)
        if ys.size == 0 or xs.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    @staticmethod
    def _prepare_reference_mask(mask, image_shape):
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape[:2] != image_shape[:2]:
            mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype(np.uint8) * 255

    @staticmethod
    def _config_bool(value, default=False):
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
        return bool(value)

    @staticmethod
    def _roll_panorama_reference_for_crop(image, mask):
        ys, xs = np.where(mask > 0)
        if ys.size == 0 or xs.size == 0:
            return image, mask

        width = mask.shape[1]
        x_unique = np.unique(xs)
        if x_unique.size <= 1:
            return image, mask

        diffs = np.diff(x_unique)
        wrap_gap = np.array([x_unique[0] + width - x_unique[-1]])
        gaps = np.concatenate([diffs, wrap_gap], axis=0)
        largest_gap_idx = int(np.argmax(gaps))
        start = int(x_unique[(largest_gap_idx + 1) % x_unique.size])
        end = int(x_unique[largest_gap_idx])
        wrapped_span = int(((end - start) % width) + 1)
        regular_span = int(x_unique[-1] - x_unique[0] + 1)
        if wrapped_span >= regular_span:
            return image, mask

        return np.roll(image, -start, axis=1), np.roll(mask, -start, axis=1)

    def _crop_reference_with_mask(self, image, mask, wrap_horizontal=False):
        mask = self._prepare_reference_mask(mask, image.shape)
        if mask is None:
            return image, None
        if wrap_horizontal:
            image, mask = self._roll_panorama_reference_for_crop(image, mask)

        bbox = self._mask_bbox_xyxy(mask)
        if bbox is None:
            return image, mask

        x1, y1, x2, y2 = bbox
        box_w = max(x2 - x1, 1)
        box_h = max(y2 - y1, 1)
        max_box_dim = max(box_w, box_h)

        fill_ratio = random.uniform(0.6, 0.9)
        crop_side = int(math.ceil(max_box_dim / max(fill_ratio, 1e-6)))
        crop_side = max(crop_side, max_box_dim, 1)
        crop_side = min(crop_side, image.shape[0], image.shape[1])

        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)

        left = int(round(center_x - crop_side / 2.0))
        top = int(round(center_y - crop_side / 2.0))
        left = max(0, min(left, image.shape[1] - crop_side))
        top = max(0, min(top, image.shape[0] - crop_side))

        return (
            image[top:top + crop_side, left:left + crop_side],
            mask[top:top + crop_side, left:left + crop_side],
        )

    def _load_reference_image_from_paths(
        self,
        image_path,
        mask_path,
        mode='train',
        wrap_horizontal=False,
        ref_entry=None,
    ):
        if image_path is None or mask_path is None:
            return None, None

        ref_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        ref_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if ref_image is None or ref_mask is None:
            return None, None

        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        ref_crop, ref_mask_crop = self._crop_reference_with_mask(
            ref_image,
            ref_mask,
            wrap_horizontal=wrap_horizontal,
        )
        if ref_mask_crop is None:
            return None, None

        ref_crop = cv2.resize(
            ref_crop,
            (self.config['refs_resolution'], self.config['refs_resolution']),
            interpolation=cv2.INTER_LINEAR,
        )
        ref_mask_crop = cv2.resize(
            ref_mask_crop,
            (self.config['refs_resolution'], self.config['refs_resolution']),
            interpolation=cv2.INTER_NEAREST,
        )

        if mode == 'train':
            ref_crop, ref_mask_crop = self.apply_refs_augmentation(ref_crop, ref_mask_crop, mode=mode)
            keep_background_prob = float(self.config.get('refs_keep_background_prob', 0.6))
            keep_background_prob = min(max(keep_background_prob, 0.0), 1.0)
            if random.random() >= keep_background_prob:
                ref_binary_mask = (ref_mask_crop > 127).astype(np.uint8)
                ref_crop = ref_crop * ref_binary_mask[:, :, None]
        elif mode in {'test', 'predict'}:
            keep_background = self._config_bool(
                self.config.get('add_ref_keep_background_in_inference', True),
                default=True,
            )
            if not keep_background:
                ref_binary_mask = (ref_mask_crop > 127).astype(np.uint8)
                ref_crop = ref_crop * ref_binary_mask[:, :, None]

        if ref_entry is None:
            ref_entry = {
                'image_path': image_path,
                'mask_path': mask_path,
            }
        return ref_crop, ref_entry

    def _sample_external_reference(self, object_id):
        object_prefix = self._extract_object_prefix(object_id)
        if object_prefix is None:
            return None

        ref_index = self._build_external_ref_index(self.external_ref_base_dir)
        prefix_index = ref_index.get(object_prefix)
        if not prefix_index:
            return None

        shuffled_saved_dirs = list(self.external_ref_saved_dirs)
        random.shuffle(shuffled_saved_dirs)
        for saved_dir in shuffled_saved_dirs:
            shuffled_cameras = list(self.external_ref_cameras)
            random.shuffle(shuffled_cameras)
            for camera_name in shuffled_cameras:
                matches = prefix_index.get(saved_dir, {}).get(camera_name, [])
                if matches:
                    return random.choice(matches)

        all_matches = []
        for saved_dir in prefix_index.values():
            for camera_matches in saved_dir.values():
                all_matches.extend(camera_matches)
        if all_matches:
            return random.choice(all_matches)
        return None

    def _load_external_reference_image(self, object_id, mode='train'):
        ref_entry = self._sample_external_reference(object_id)
        if ref_entry is None:
            return None, None

        return self._load_reference_image_from_paths(
            ref_entry['image_path'],
            ref_entry['mask_path'],
            mode=mode,
            wrap_horizontal=False,
            ref_entry=ref_entry,
        )

    def _load_self_reference_image(self, image_path, mask_path, mode='train'):
        return self._load_reference_image_from_paths(
            image_path,
            mask_path,
            mode=mode,
            wrap_horizontal=True,
            ref_entry={
                'image_path': image_path,
                'mask_path': mask_path,
                'source_id': 'self_crop',
            },
        )
    
    @abstractmethod
    def get_data(self, idx):
        pass

    def __getitem__(self, idx):
        data = self.get_data(idx)
        self.random_clip = random.random()
        sampled_ref = None
        sampled_ref_entry = None

        

        # generate camera poses
        if self.config['cam_sampler'] == 'horizon':
            theta, phi = horizon_sample_camera(8)
            if self.mode == 'train':
                cam_rot = random.random() * 360
                theta = (theta + cam_rot) % 360
                if 'prompt' in data:
                    shift_idx = round(cam_rot / 45)
                    data['prompt'] = data['prompt'][shift_idx:] + data['prompt'][:shift_idx]
        elif self.config['cam_sampler'] == 'icosahedron':
            if self.mode == 'train':
                if self.use_fixed_pers_prompt==True and self.use_cubemap_prompt==False:
                    theta, phi = icosahedron_sample_camera()#random_sample_camera(20) #20
                elif self.use_fixed_pers_prompt==True and self.use_cubemap_prompt==True:
                    theta, phi = cubemap_sample_camera()
                else:
                    theta, phi = random_sample_camera(20)
            elif self.mode == 'val' and self.use_cubemap_prompt==True:
                theta, phi = cubemap_sample_camera()
            else:
                theta, phi = cubemap_sample_camera()#icosahedron_sample_camera()
        else:
            raise NotImplementedError
        theta, phi = np.rad2deg(theta), np.rad2deg(phi) # Used to convert radians to degrees

        Ks, Rs = [], []
        for t, p in zip(theta, phi):
            K, R = get_K_R(self.config['fov'], t, p,
                           self.config['pers_resolution'], self.config['pers_resolution'])
            Ks.append(K)
            Rs.append(R)
        K = np.stack(Ks).astype(np.float32)
        R = np.stack(Rs).astype(np.float32)

        cameras = {
            'height': np.full_like(theta, self.config['pers_resolution'], dtype=int),
            'width': np.full_like(theta, self.config['pers_resolution'], dtype=int),
            'FoV': np.full_like(theta, self.config['fov'], dtype=int),
            'theta': theta,
            'phi': phi,
            'R': R,
            'K': K,
        }
        data['cameras'] = cameras
        data['height'] = self.config['pano_height']
        data['width'] = self.config['pano_height'] * 2

        if 'rotation_type' in data:
            if data['rotation_type'] == 'relative':
                rotation = random.random() * 360 if self.mode == 'train' and self.config['rand_rot_img'] else 0
            elif data['rotation_type'] == 'absolute':
                rotation = random.random() * 10 if self.mode == 'train' and self.config['rand_rot_img'] else 0
        else:
            rotation = 0
        # flip = self.config['rand_flip'] and self.mode == 'train' and random.random() < 0.5
        flip = False

    
        def process_equi(equi, normalize, mode='train', function='add'):
            imgs = []
            refs = []
            images = []
            test_pers = []
            ref = sampled_ref if function in {'add'} else None

            if ref is not None and function in {'add'}:
                if ref.shape[:2] != (self.config['refs_resolution'], self.config['refs_resolution']):
                    ref = cv2.resize(ref, (self.config['refs_resolution'], self.config['refs_resolution']), 
                                   interpolation=cv2.INTER_LINEAR)
                refs.append(ref)
                refs = np.stack(refs)
                refs = (refs.astype(np.float32)/127.5)-1
                refs = rearrange(refs, 'b h w c -> b c h w')

            else:
                empty_ref = np.zeros((1, 3, self.config['refs_resolution'], self.config['refs_resolution']), dtype=np.float32)
                refs = empty_ref
            if self.use_fixed_pers_prompt==True and self.use_cubemap_prompt==True and self.only_pano==False:
                assert self.only_pano==False, "only_pano must be False when use_fixed_pers_prompt and use_cubemap_prompt are True"
                equi.rotate(rotation)
                equi.flip(flip)
                
                for t, p in zip(theta, phi):
                    img = equi.to_perspective((self.config['fov'], self.config['fov']), t, p, (self.config['pers_resolution'], self.config['pers_resolution']))
                    imgs.append(img)
                images = np.stack(imgs)
                if self.result_dir is None and normalize:
                    images = (images.astype(np.float32)/127.5)-1

                images = rearrange(images, 'b h w c -> b c h w')

            elif self.only_pano==True and self.use_fixed_pers_prompt==False and self.use_cubemap_prompt==False:
                equi.rotate(rotation)
                equi.flip(flip)

            pano = cv2.resize(equi.equirectangular, (data['width'], data['height']), interpolation=cv2.INTER_AREA)
            pano = pano.reshape(data['height'], data['width'], equi.equirectangular.shape[-1])
            if self.result_dir is None and normalize:
                pano = (pano.astype(np.float32)/127.5 - 1)
            pano = rearrange(pano, 'h w c -> 1 c h w')

            return pano, images, refs
        

        def process_mask_equi(mask_path, normalize, binarize=True):
            if mask_path is None or not os.path.exists(mask_path):
                if mask_path is not None:
                    print(f"Warning: Cannot read mask file: {mask_path}, using zero mask")
                mask_img = np.zeros((data['height'], data['width']), dtype=np.float32)
            else:
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is None:
                    print(f"Warning: Cannot read mask file: {mask_path}, using zero mask")
                    mask_img = np.zeros((data['height'], data['width']), dtype=np.float32)
                else:
                    interpolation = cv2.INTER_NEAREST if binarize else cv2.INTER_LINEAR
                    mask_img = cv2.resize(mask_img, (data['width'], data['height']), interpolation=interpolation)
                    if binarize:
                        # Binarize mask (ensure only 0 and 1)
                        mask_img = (mask_img > 128).astype(np.float32)
                    else:
                        mask_img = mask_img.astype(np.float32)
                        if mask_img.max() > 1.0:
                            mask_img = mask_img / 255.0
                        mask_img = np.clip(mask_img, 0.0, 1.0)

            # Create Equirectangular object for generating perspective view
            equi = Equirectangular(mask_img[:, :, None])  # Add channel dimension
            equi.rotate(rotation)
            equi.flip(flip)
            pano_mask = equi.equirectangular

            # Convert to expected format
            pano_mask = pano_mask[None,  :, :, :].copy()  # [1, 1, h, w]
            pano_mask = rearrange(pano_mask, 'b h w c -> b c h w')
            
            return pano_mask

        def process_rgb_equi(image_path):
            if image_path is None or not os.path.exists(image_path):
                return None

            equi = Equirectangular.from_file(image_path)
            equi.rotate(rotation)
            equi.flip(flip)
            pano = cv2.resize(
                equi.equirectangular,
                (data['width'], data['height']),
                interpolation=cv2.INTER_AREA,
            )
            pano = pano.reshape(data['height'], data['width'], equi.equirectangular.shape[-1])
            if pano.ndim == 2:
                pano = pano[:, :, None]
            if pano.shape[2] == 1:
                pano = np.repeat(pano, 3, axis=2)
            elif pano.shape[2] > 3:
                pano = pano[:, :, :3]

            pano = pano.astype(np.float32)
            if pano.max() <= 1.0 and pano.min() >= 0.0:
                pano = pano * 255.0
            return np.clip(pano, 0.0, 255.0).astype(np.float32)

        def process_depth_equi(depth_path, source_rgb_path=None):
            if depth_path is None or not os.path.exists(depth_path):
                raise FileNotFoundError(
                    "Missing depth conditioning file for input panorama. "
                    f"rgb_path={source_rgb_path}, expected_depth_path={depth_path}"
                )

            try:
                depth = np.load(depth_path)
            except Exception as exc:
                raise RuntimeError(f"Cannot read depth conditioning file: {depth_path}") from exc

            try:
                depth = self._normalize_depth_for_vae(
                    depth,
                    lower_quantile=self.config.get('depth_cond_lower_quantile', 0.0),
                    upper_quantile=self.config.get('depth_cond_upper_quantile', 0.98),
                    use_log=bool(self.config.get('depth_cond_use_log', False)),
                )
            except ValueError as exc:
                raise ValueError(f"Invalid depth conditioning file: {depth_path}") from exc

            equi = Equirectangular(depth[:, :, None])
            equi.rotate(rotation)
            equi.flip(flip)
            depth_pano = cv2.resize(
                equi.equirectangular,
                (data['width'], data['height']),
                interpolation=cv2.INTER_LINEAR,
            )
            depth_pano = depth_pano.reshape(data['height'], data['width'], 1).astype(np.float32)
            depth_pano = np.repeat(depth_pano, 3, axis=2)
            depth_pano = depth_pano * 2.0 - 1.0
            depth_pano = rearrange(depth_pano, 'h w c -> 1 c h w')
            return depth_pano

        def process_gt_depth_equi(depth_path, source_rgb_path=None):
            if depth_path is None or not os.path.exists(depth_path):
                raise FileNotFoundError(
                    "Missing GT scene-depth file for target panorama. "
                    f"rgb_path={source_rgb_path}, expected_depth_path={depth_path}"
                )

            try:
                depth = self._read_exr_depth(depth_path)
            except Exception as exc:
                raise RuntimeError(f"Cannot read GT scene-depth file: {depth_path}") from exc

            equi = Equirectangular(depth[:, :, None])
            equi.rotate(rotation)
            equi.flip(flip)
            depth_pano = cv2.resize(
                equi.equirectangular,
                (data['width'], data['height']),
                interpolation=cv2.INTER_LINEAR,
            )
            depth_pano = depth_pano.reshape(data['height'], data['width'], 1).astype(np.float32)
            depth_pano = rearrange(depth_pano, 'h w c -> 1 c h w')
            return depth_pano

        if self.use_ref:
            function = str(data.get('function', '')).lower()
            if function == 'add':
                provided_ref_image_path = data.get('reference_image_path') or data.get('ref_img_path')
                provided_ref_mask_path = data.get('reference_mask_path') or data.get('ref_mask_path')
                if provided_ref_image_path and provided_ref_mask_path:
                    sampled_ref, sampled_ref_entry = self._load_reference_image_from_paths(
                        provided_ref_image_path,
                        provided_ref_mask_path,
                        mode=self.mode,
                        wrap_horizontal=False,
                        ref_entry={
                            'image_path': provided_ref_image_path,
                            'mask_path': provided_ref_mask_path,
                            'source_id': 'sample_reference',
                        },
                    )
                if sampled_ref is None:
                    sampled_ref, sampled_ref_entry = self._load_dino_reference_image(
                        data.get('pano_path'),
                        mode=self.mode,
                    )
                if sampled_ref is None and self.dino_ref_fallback_to_external:
                    sampled_ref, sampled_ref_entry = self._load_external_reference_image(
                        data.get('object_id'),
                        mode=self.mode,
                    )

        data['has_ref'] = sampled_ref is not None
        if sampled_ref_entry is not None:
            data['ref_img_path'] = sampled_ref_entry['image_path']
            data['ref_mask_path'] = sampled_ref_entry['mask_path']
            data['ref_source_id'] = sampled_ref_entry.get('source_id')
            if data['ref_source_id'] is None:
                saved_dir = sampled_ref_entry.get('saved_dir')
                camera_name = sampled_ref_entry.get('camera_name')
                object_name = sampled_ref_entry.get('object_name')
                if saved_dir is not None and camera_name is not None and object_name is not None:
                    data['ref_source_id'] = f"{saved_dir}/{camera_name}/{object_name}"
                else:
                    data['ref_source_id'] = os.path.basename(sampled_ref_entry['image_path'])
        
        # load images
        if 'pano_path' in data and 'remove_pano_path' in data:
            target_pano_path = data.get('pano_path')
            input_pano_path = data.get('remove_pano_path')

            if target_pano_path is None:
                if input_pano_path is None:
                    raise FileNotFoundError(
                        "Both `pano_path` and `remove_pano_path` are missing; cannot build an inference sample."
                    )
                # Pure inference datasets may omit the GT `after` panorama. Reuse the
                # edit input as a placeholder target so downstream code still receives
                # tensors with the expected shapes for saving and visualization.
                target_pano_path = input_pano_path

            equirectangular = Equirectangular.from_file(target_pano_path)
            data['pano'],data['images'],data['refs'] = process_equi(
                equirectangular, True, mode=self.mode, function=data['function']
            )

            if input_pano_path is not None:
                equirectangular = Equirectangular.from_file(input_pano_path)
                data['remove_pano'],data['remove_images'], _ = process_equi(
                    equirectangular, True, mode=self.mode, function=data['function']
                )
            else:
                # Some demo-style inference inputs may only provide a target pano.
                data['remove_pano'] = data['pano'].copy()
                data['remove_images'] = data['images'].copy() if len(data['images']) > 0 else []

            input_depth_rgb_path = self._resolve_input_condition_rgb_path(data)
            input_depth_path = self._resolve_depth_path_from_rgb_path(input_depth_rgb_path)
            data['input_depth_path'] = input_depth_path
            data['input_depth'] = process_depth_equi(input_depth_path, input_depth_rgb_path)

            if self.mode == 'train' and bool(self.config.get('load_gt_depth', True)):
                if data['function'] == 'remove':
                    target_depth_rgb_path = input_pano_path
                else:
                    target_depth_rgb_path = target_pano_path
                target_depth_path = self._resolve_gt_depth_path_from_rgb_path(
                    target_depth_rgb_path,
                    depth_root_name=self.config.get(
                        'gt_depth_root_name',
                        'MovieRenders_SceneDepth_1024_gt',
                    ),
                )
                data['target_depth_path'] = target_depth_path
                data['target_depth'] = process_gt_depth_equi(
                    target_depth_path,
                    target_depth_rgb_path,
                )

            full_mask_path1 = data.get('full_mask_path1')
            full_mask_path2 = data.get('full_mask_path2')
            guidance_mask_path1 = self._select_guidance_mask_path(
                data.get('pano_mask_path1'),
                full_mask_path1,
            )
            guidance_mask_path2 = self._select_source_guidance_mask_path(
                data.get('pano_mask_path2'),
                full_mask_path2,
            )
            point_mask_path1 = self._resolve_point_mask_path(full_mask_path1 or guidance_mask_path1)
            pano_mask1 = process_mask_equi(guidance_mask_path1, True)
            pano_mask2 = process_mask_equi(guidance_mask_path2, True)
            pano_full_mask1 = process_mask_equi(full_mask_path1 or guidance_mask_path1, True)
            pano_full_mask2 = process_mask_equi(full_mask_path2 or guidance_mask_path2, True)
            pano_point_mask1 = process_mask_equi(point_mask_path1, True, binarize=False)
            # 3通道mask: 通道0=目标位置/after(mask1), 通道1=源位置/before(mask2), 通道2=预生成target point
            # 移动: 两通道都有值; 添加: 只有通道0; 删除: 只有通道1
            data['pano_mask'] = np.concatenate([pano_mask1, pano_mask2, pano_point_mask1], axis=1)  # [1, 3, H, W]
            data['pano_full_mask'] = np.concatenate([pano_full_mask1, pano_full_mask2, pano_point_mask1], axis=1)
            data['guidance_mask_path1'] = guidance_mask_path1
            data['guidance_mask_path2'] = guidance_mask_path2
            data['guidance_point_mask_path1'] = point_mask_path1
            data['inference_source_mask_type'] = self._normalize_source_mask_type(
                self.config.get('inference_source_mask_type', 'bbox')
            )



        #flip the perspective image prompt
        if flip:
            data['prompt'] = data['prompt'][::-1]

        # load pano prompt
        if 'pano_prompt' not in data:
            if data['function'] == 'move':
                data['pano_prompt'] = self.load_prompt_move(data['pano_prompt_path'])
                data['pano_prompt_without_img'] = self.load_prompt_without_img_move(data['pano_prompt_path'])
                data['pano_prompt_with_mask'] = self.load_prompt_move_with_mask(data['pano_prompt_path'])
                data['pano_prompt_with_ref'] = data['pano_prompt']
                data['pano_prompt_with_ref_and_mask'] = data['pano_prompt_with_mask']
            elif data['function'] == 'remove':
                data['pano_prompt'] = self.load_prompt_remove(data['pano_prompt_path'])
                data['pano_prompt_without_img'] = self.load_prompt_without_img_remove(data['pano_prompt_path'])
                data['pano_prompt_with_mask'] = self.load_prompt_remove_with_mask(data['pano_prompt_path'])
                data['pano_prompt_with_ref'] = data['pano_prompt']
                data['pano_prompt_with_ref_and_mask'] = data['pano_prompt_with_mask']
            else:
                data['pano_prompt'] = self.load_prompt_add(data['pano_prompt_path'])
                data['pano_prompt_without_img'] = self.load_prompt_without_img_add(data['pano_prompt_path'])
                data['pano_prompt_with_mask'] = self.load_prompt_add_with_mask(data['pano_prompt_path'])
                data['pano_prompt_with_ref'] = self.load_prompt_with_ref(data['pano_prompt_path'])
                data['pano_prompt_with_ref_and_mask'] = self.load_prompt_with_ref_and_mask(data['pano_prompt_path'])

        if 'pano_simple_prompt' not in data and self.mode != 'predict' and self.mode != 'test':
            if data['function'] == 'move':
                data['pano_simple_prompt'] = self.load_prompt_move(data['pano_simple_prompt_path'])
                data['pano_simple_prompt_without_img'] = self.load_prompt_without_img_move(data['pano_simple_prompt_path'])
                data['pano_simple_prompt_with_mask'] = self.load_prompt_move_with_mask(data['pano_simple_prompt_path'])
                data['pano_simple_prompt_with_ref'] = data['pano_simple_prompt']
                data['pano_simple_prompt_with_ref_and_mask'] = data['pano_simple_prompt_with_mask']
            elif data['function'] == 'remove':
                data['pano_simple_prompt'] = self.load_prompt_remove(data['pano_simple_prompt_path'])
                data['pano_simple_prompt_without_img'] = self.load_prompt_without_img_remove(data['pano_simple_prompt_path'])
                data['pano_simple_prompt_with_mask'] = self.load_prompt_remove_with_mask(data['pano_simple_prompt_path'])
                data['pano_simple_prompt_with_ref'] = data['pano_simple_prompt']
                data['pano_simple_prompt_with_ref_and_mask'] = data['pano_simple_prompt_with_mask']
            else:
                data['pano_simple_prompt'] = self.load_prompt_add(data['pano_simple_prompt_path'])
                data['pano_simple_prompt_without_img'] = self.load_prompt_without_img_add(data['pano_simple_prompt_path'])
                data['pano_simple_prompt_with_mask'] = self.load_prompt_add_with_mask(data['pano_simple_prompt_path'])
                data['pano_simple_prompt_with_ref'] = self.load_prompt_with_ref(data['pano_simple_prompt_path'])
                data['pano_simple_prompt_with_ref_and_mask'] = self.load_prompt_with_ref_and_mask(data['pano_simple_prompt_path'])

        if self.mode == 'test' and 'test_pers_prompt_path' in data:
            data['test_pers_prompt'] = self.load_test_pers_prompt(data['test_pers_prompt_path'])
        # load forward prompt
        # if 'forward_prompt' not in data:
        #     data['forward_prompt'] = self.load_prompt(data['forward_prompt_path'])
        # # load reverse prompt
        # if 'reverse_prompt' not in data:
        #     data['reverse_prompt'] = self.load_prompt(data['reverse_prompt_path'])
        # # load edited prompt
        # if 'edited_prompt' not in data:
        #     data['edited_prompt'] = self.load_prompt(data['edited_prompt_path'])
        if self.mode == 'train' and self.result_dir is None and random.random() < self.config['simple_prompt_ratio']:
            data['pano_prompt'] = data['pano_simple_prompt']
            if 'pano_simple_prompt_with_mask' in data:
                data['pano_prompt_with_mask'] = data['pano_simple_prompt_with_mask']
            if 'pano_simple_prompt_with_ref' in data:
                data['pano_prompt_with_ref'] = data['pano_simple_prompt_with_ref']
            if 'pano_simple_prompt_with_ref_and_mask' in data:
                data['pano_prompt_with_ref_and_mask'] = data['pano_simple_prompt_with_ref_and_mask']
        # unconditioned training
        if self.mode == 'train' and self.result_dir is None and random.random() < self.config['conditioning_dropout_prob']:
            data['pano_prompt'] = ''
            # if 'prompt' in data:
            #     data['prompt'] = [''] * len(data['prompt'])
        if self.mode == 'train' and self.result_dir is None and random.random() < self.config['uncond_ratio_pers']:
            data['prompt'] = [''] * len(data['prompt'])
        # load results
        if self.config['gt_as_result']:
            if data.get('function') == 'remove':
                data['pano_pred'] = data['remove_pano']
                data['images_pred'] = data['remove_images']
            else:
                data['pano_pred'] = data['pano']
                data['images_pred'] = data['images']
            # data['pano_pred'] = rearrange(data['pano_pred'], '1 c h w -> 1 h w c')
        elif self.result_dir is not None:
            images_pred = []
            for i in range(len(data['images'])):
                image_pred_path = os.path.join(os.path.dirname(data['pano_pred_path']), f"{i}.png")
                if not os.path.exists(image_pred_path):
                    break
                image_pred = Image.open(image_pred_path).convert('RGB')
                image_pred = np.array(image_pred)
                image_pred = cv2.resize(image_pred, (self.config['pers_resolution'], self.config['pers_resolution']))
                images_pred.append(image_pred)
            if images_pred:
                images_pred = np.stack(images_pred)
                data['images_pred'] = rearrange(images_pred, 'b h w c -> b c h w')

            if os.path.exists(data['pano_pred_path']):
                equirectangular = Equirectangular.from_file(data['pano_pred_path'])
                pano = cv2.resize(equirectangular.equirectangular, (data['width'], data['height']))
                pano = pano.reshape(data['height'], data['width'], equirectangular.equirectangular.shape[-1])
                data['pano_pred'] = rearrange(pano, 'h w c -> 1 c h w')
                if data.get('json_path') is not None:
                    json_data = json.load(open(data['json_path']))
                    transform_params = json_data.get("perspective_transform_params", {})
                    center_u_deg = transform_params.get("final_center_u_deg")
                    center_v_deg = transform_params.get("final_center_v_deg") 
                    hfov_deg = transform_params.get("final_hfov_deg")
                    vfov_deg = transform_params.get("final_vfov_deg")
                    test_pers_resolution = (self.config['pers_resolution'], self.config['pers_resolution'])
                    test_pers = equirectangular.to_perspective((hfov_deg, vfov_deg), center_u_deg, center_v_deg, test_pers_resolution)
                    # test_pers = (test_pers.astype(np.float32)/127.5)-1
                    test_pers = rearrange(test_pers, 'h w c -> 1 c h w')
                    data['test_pers'] = test_pers
                if self.test_function == 'remove':
                    ori_equirectangular = Equirectangular.from_file(data['remove_pano_path'])
                else:
                    ori_equirectangular = Equirectangular.from_file(data['pano_path'])
                ori_pano = ori_equirectangular.equirectangular
                ori_pano = cv2.resize(ori_pano, (data['width'], data['height']))
                ori_pano = ori_pano.reshape(data['height'], data['width'], ori_pano.shape[-1])
                data['pano'] = rearrange(ori_pano, 'h w c -> 1 c h w')
                if data.get('json_path') is not None:
                    json_data = json.load(open(data['json_path']))
                    transform_params = json_data.get("perspective_transform_params", {})
                    center_u_deg = transform_params.get("final_center_u_deg")
                    center_v_deg = transform_params.get("final_center_v_deg") 
                    hfov_deg = transform_params.get("final_hfov_deg")
                    vfov_deg = transform_params.get("final_vfov_deg")
                    test_pers_resolution = (self.config['pers_resolution'], self.config['pers_resolution'])
                    test_pers = ori_equirectangular.to_perspective((hfov_deg, vfov_deg), center_u_deg, center_v_deg, test_pers_resolution)
                    # test_pers = (test_pers.astype(np.float32)/127.5)-1
                    test_pers = rearrange(test_pers, 'h w c -> 1 c h w')
                    data['ori_test_pers'] = test_pers
            elif 'images_pred' in data:
                # merge images for MVDiffusion results
                pano = mp2e(
                    images_pred, cameras['FoV'], cameras['theta'], cameras['phi'],
                    (data['height'], data['width']))
                data['pano_pred'] = rearrange(pano, 'h w c -> 1 c h w')

        return self._build_mover360_batch_output(data)

    def apply_refs_augmentation(self, ref_img, ref_mask=None, mode='train'):
        if ref_mask is not None:
            if ref_mask.ndim == 3:
                ref_mask = ref_mask[:, :, 0]
            ref_mask = (ref_mask > 127).astype(np.uint8) * 255

        if mode != 'train' or not self.config.get('refs_augmentation', True):
            if ref_mask is None:
                return ref_img
            return ref_img, ref_mask
        
        augmented_ref = ref_img.copy()
        augmented_mask = None if ref_mask is None else ref_mask.copy()
        
        # 1. random flip
        if random.random() < self.config.get('refs_flip_prob', 0.5):
            augmented_ref = np.flip(augmented_ref, axis=1).copy()
            if augmented_mask is not None:
                augmented_mask = np.flip(augmented_mask, axis=1).copy()
        
        # 2. random rotation (-15° to +15°)
        if random.random() < self.config.get('refs_rotation_prob', 0.7):
            angle = random.uniform(-15, 15)
            h, w = augmented_ref.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented_ref = cv2.warpAffine(augmented_ref, M, (w, h), 
                                         borderMode=cv2.BORDER_REFLECT_101)
            if augmented_mask is not None:
                augmented_mask = cv2.warpAffine(
                    augmented_mask,
                    M,
                    (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
        
        # 3. random affine transformation (slight perspective transformation)
        if random.random() < self.config.get('refs_affine_prob', 0.3):
            h, w = augmented_ref.shape[:2]
            pts1 = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], dtype=np.float32)
            # add random offset (maximum offset is 3% of image size, reduce offset to avoid excessive deformation)
            offset = min(w, h) * 0.03
            # ensure the points after offset are still reasonable
            offsets = np.random.uniform(-offset, offset, (4, 2)).astype(np.float32)
            pts2 = pts1 + offsets
            
            # ensure the points after transformation are within reasonable range
            pts2[:, 0] = np.clip(pts2[:, 0], -w*0.1, w*1.1)
            pts2[:, 1] = np.clip(pts2[:, 1], -h*0.1, h*1.1)
            
            try:
                M = cv2.getPerspectiveTransform(pts1, pts2)
                augmented_ref = cv2.warpPerspective(augmented_ref, M, (w, h),
                                                  borderMode=cv2.BORDER_REFLECT_101)
                if augmented_mask is not None:
                    augmented_mask = cv2.warpPerspective(
                        augmented_mask,
                        M,
                        (w, h),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
            except cv2.error:
                # if perspective transformation fails, skip this augmentation
                print("perspective transformation failed")
                pass
        
        # 4. random scale and crop
        if random.random() < self.config.get('refs_scale_prob', 0.5):
            h, w = augmented_ref.shape[:2]
            scale = random.uniform(0.9, 1.1)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # scale image
            augmented_ref = cv2.resize(augmented_ref, (new_w, new_h), 
                                     interpolation=cv2.INTER_LINEAR)
            if augmented_mask is not None:
                augmented_mask = cv2.resize(
                    augmented_mask,
                    (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            
            # if the scaled size is larger than the original size, perform center crop
            if new_h > h or new_w > w:
                start_y = (new_h - h) // 2
                start_x = (new_w - w) // 2
                augmented_ref = augmented_ref[start_y:start_y+h, start_x:start_x+w]
                if augmented_mask is not None:
                    augmented_mask = augmented_mask[start_y:start_y+h, start_x:start_x+w]
            # if the scaled size is smaller than the original size, perform padding
            elif new_h < h or new_w < w:
                pad_y = (h - new_h) // 2
                pad_x = (w - new_w) // 2
                augmented_ref = np.pad(augmented_ref, 
                                     ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x), (0, 0)),
                                     mode='reflect')
                if augmented_mask is not None:
                    augmented_mask = np.pad(
                        augmented_mask,
                        ((pad_y, h-new_h-pad_y), (pad_x, w-new_w-pad_x)),
                        mode='constant',
                        constant_values=0,
                    )
        
        # 5. random color enhancement
        if random.random() < self.config.get('refs_color_prob', 0.6):
            # brightness adjustment
            brightness_factor = random.uniform(0.8, 1.2)
            augmented_ref = np.clip(augmented_ref * brightness_factor, 0, 255)
            
                # contrast adjustment
            if random.random() < 0.5:
                contrast_factor = random.uniform(0.8, 1.2)
                mean_val = np.mean(augmented_ref)
                augmented_ref = np.clip((augmented_ref - mean_val) * contrast_factor + mean_val, 0, 255)
            
            # saturation adjustment
            if random.random() < 0.5 and len(augmented_ref.shape) == 3:
                saturation_factor = random.uniform(0.8, 1.2)
                # convert to HSV space and adjust saturation
                try:
                    hsv = cv2.cvtColor(augmented_ref.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
                    augmented_ref = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
                except cv2.error:
                    # if color space conversion fails, skip saturation adjustment
                    pass
        
        # 6. random gaussian noise
        if random.random() < self.config.get('refs_noise_prob', 0.3):
            noise_std = random.uniform(1, 5)
            noise = np.random.normal(0, noise_std, augmented_ref.shape)
            augmented_ref = np.clip(augmented_ref + noise, 0, 255)
        
        # 7. random gaussian blur
        if random.random() < self.config.get('refs_blur_prob', 0.2):
            kernel_size = random.choice([3, 5])
            augmented_ref = cv2.GaussianBlur(augmented_ref, (kernel_size, kernel_size), 0)
        
        # 8. random sharpening
        if random.random() < self.config.get('refs_sharpen_prob', 0.2):
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
            sharpened = cv2.filter2D(augmented_ref, -1, kernel)
            # mix original image and sharpened image
            sharpen_factor = random.uniform(0.1, 0.3)
            augmented_ref = cv2.addWeighted(augmented_ref, 1-sharpen_factor, sharpened, sharpen_factor, 0)
        
        # 9. random gamma correction
        if random.random() < self.config.get('refs_gamma_prob', 0.3):
            gamma = random.uniform(0.7, 1.3)
            # normalize image to [0,1], apply gamma correction, then restore to [0,255]
            augmented_ref = augmented_ref / 255.0
            augmented_ref = np.power(np.clip(augmented_ref, 0, 1), gamma)
            augmented_ref = augmented_ref * 255.0
            augmented_ref = np.clip(augmented_ref, 0, 255)
        
        # 10. random block crop
        if random.random() < self.config.get('refs_block_crop_prob', 0.4):
            h, w = augmented_ref.shape[:2]
            # randomly generate 1-5 blocks
            num_blocks = random.randint(1, 5)
            
            for _ in range(num_blocks):
                # block size is 5%-15% of image size
                block_size_ratio = random.uniform(0.05, 0.15)
                block_h = int(h * block_size_ratio)
                block_w = int(w * block_size_ratio)
                
                # ensure the block size is at least 5x5 pixels
                block_h = max(5, block_h)
                block_w = max(5, block_w)
                
                # randomly select block position
                start_y = random.randint(0, max(0, h - block_h))
                start_x = random.randint(0, max(0, w - block_w))
                
                # fill block area with black
                augmented_ref[start_y:start_y+block_h, start_x:start_x+block_w] = 0
        
        augmented_ref = augmented_ref.astype(np.uint8)
        if augmented_mask is None:
            return augmented_ref
        augmented_mask = (augmented_mask > 127).astype(np.uint8) * 255
        return augmented_ref, augmented_mask


class PanoDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str = None,
            fov: int = 95,
            cam_sampler: str = 'icosahedron',  # 'horizon', 'icosahedron'
            refs_resolution: int = 384, # the size of reference image
            pers_resolution: int = 256,
            pano_height: int = 512, 
            # uncond_ratio: float = 0.2,
            conditioning_dropout_prob: float = 0.01,
            simple_prompt_ratio: float = 0.3,
            uncond_ratio_pers: float = 0,
            train_batch_size: int = 2,
            val_batch_size: int = 1,
            num_workers: int = 2,
            result_dir: str = None,
            rand_rot_img: bool = True,
            rand_flip: bool = False,
            gt_as_result: bool = False,
            repeat_predict: int = 1,
            only_pano: bool = True,
            use_fixed_pers_prompt: bool = False,
            use_cubemap_prompt: bool = False,
            test_function: str = 'move', #remove
            # Flat-scan remove entries edit the BEFORE pano at the source-object
            # mask instead of the AFTER pano at the target mask. Use for the
            # move chain's stage-1 removal: the default remove definition edits
            # the after pano, which leaks GT after-state into move backgrounds.
            remove_from_before: bool = False,
            # Iterate predict/test entries back-to-front. Lets a second worker
            # process share one predict run with the primary (skip-existing
            # splits the work as the two fronts converge) instead of
            # lockstep-duplicating every sample.
            predict_reverse: bool = False,
            use_txt_prompt: bool = False, # read prompt txt files; False uses predefined task templates
            add_ref_keep_background_in_inference: bool = True,  # add predict/test reference keeps background by default
            inference_source_mask_type: str = 'bbox',  # bbox or full source mask for val/test/predict
            guidance_full_mask_prob: float = 0.5,
            depth_cond_lower_quantile: float = 0.0,
            depth_cond_upper_quantile: float = 0.98,
            depth_cond_use_log: bool = False,
            load_gt_depth: bool = False,
            gt_depth_root_name: str = 'MovieRenders_SceneDepth_1024_gt',

            use_ref: bool = True,
            refs_augmentation: bool = True,      # whether to enable refs data augmentation
            refs_flip_prob: float = 0,        # horizontal flip probability
            refs_rotation_prob: float = 0.2,    # rotation probability
            refs_affine_prob: float = 0,      # affine transformation probability
            refs_scale_prob: float = 0.5,       # scale probability
            refs_color_prob: float = 0.1,       # color enhancement probability
            refs_noise_prob: float = 0.1,       # noise addition probability
            refs_blur_prob: float = 0.1,        # blur probability
            refs_sharpen_prob: float = 0.1,     # sharpening probability
            refs_gamma_prob: float = 0.1,       # gamma correction probability
            refs_block_crop_prob: float = 0,  # random block crop probability
            refs_keep_background_prob: float = 0.4,  # keep background probability; otherwise zero the ref background
            external_ref_base_dir: str = 'data/UE5_data/Scene_pers1',
            dino_ref_json_path: str = DEFAULT_DINO_REF_JSON_PATH,
            dino_ref_selection: str = 'random',
            dino_ref_fallback_to_external: bool = True,
            dino_ref_strict: bool = True,
            ):
        super().__init__()
        self.save_hyperparameters()
        self.result_dir = result_dir
        self.dataset_cls = PanoDataset

    def _build_dataset_config(self):
        return {key: value for key, value in self.hparams.items()}

    def setup(self, stage=None):
        if self.result_dir is not None:
            self.hparams['result_dir'] = self.result_dir

        dataset_config = self._build_dataset_config()
        
        if stage in ('fit', None):
            self.train_dataset = self.dataset_cls(dataset_config, mode='train')

        if stage in ('fit', 'validate', None):
            self.val_dataset = self.dataset_cls(dataset_config, mode='val')

        if stage in ('test', None):
            self.test_dataset = self.dataset_cls(dataset_config, mode='test')

        if stage in ('predict', None):
            self.predict_dataset = self.dataset_cls(dataset_config, mode='predict')


        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.hparams.train_batch_size,
            shuffle=True, num_workers=self.hparams.num_workers, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.hparams.val_batch_size,
            shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.hparams.val_batch_size,
            shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset, batch_size=self.hparams.val_batch_size,
            shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)
