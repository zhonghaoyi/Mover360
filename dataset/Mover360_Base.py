from __future__ import annotations
import os
import numpy as np
import random
import warnings
from glob import glob
from .PanoDataset import PanoDataset, PanoDataModule

class Mover360Dataset(PanoDataset):
    _PREDICT_FUNCTIONS = {'add', 'remove', 'move'}

    def __init__(self, config, mode='train'):
        # Call parent constructor first, but temporarily do not set result_dir
        original_result_dir = config.get('result_dir')
        self._initial_result_dir = original_result_dir
        self.test_function = str(config.get('test_function', 'move')).lower()
        self.remove_from_before = bool(config.get('remove_from_before', False))
        if original_result_dir is not None:
            config = config.copy()
            config['result_dir'] = None
        
        super().__init__(config, mode)
        
        # If result_dir exists, apply custom filtering logic
        if original_result_dir is not None:
            self.result_dir = original_result_dir
            results = self.scan_results(self.result_dir)
            assert results, f"No results found in {self.result_dir}, forgot to set environment variable WANDB_RUN_ID?"
            
            results_set = set(results)
            new_data = []
            for d in self.data:
                result_key = self._build_result_key(d)
                if result_key is None:
                    continue
                if result_key in results_set:
                    d['result_dir_name'] = result_key
                    new_data.append(d)
                    continue

                matching_dirs = [r for r in results_set if result_key in r]
                if matching_dirs:
                    d['result_dir_name'] = matching_dirs[0]
                    new_data.append(d)
            
            if len(new_data) != len(self.data):
                print(f"WARNING: {len(self.data)-len(new_data)} views are missing in results folder {self.result_dir} for {self.mode} set.")
                self.data = list(new_data)
                self.data.sort(
                    key=lambda entry: (
                        entry.get('result_dir_name', ''),
                        self._build_result_key(entry) or '',
                    )
                )
        else:
            self.result_dir = None

        # Second-worker mode: iterate entries back-to-front so a helper
        # process converges with the primary instead of duplicating it.
        if bool(config.get('predict_reverse', False)) and mode in ('predict', 'test'):
            self.data = list(self.data)[::-1]
            print(f"predict_reverse: iterating {len(self.data)} {mode} entries back-to-front")

    @staticmethod
    def _extract_frame_index(filename):
        frame_stem = os.path.splitext(filename)[0]
        frame_suffix = frame_stem.rsplit('_', 1)[-1]
        try:
            return int(frame_suffix)
        except ValueError:
            return None

    @staticmethod
    def _build_result_key(entry):
        if 'pano_id' in entry:
            return entry['pano_id']
        if 'view_id' in entry:
            return entry['view_id']

        function = entry.get('function')
        if entry.get('predict_style_sample', False):
            result_key = entry.get('pano_id_base')
            if result_key is None:
                scene_id = entry.get('scene_id')
                sample_id = entry.get('sample_id', entry.get('object_id'))
                frame_id = entry.get('frame_id')
                if scene_id is not None and sample_id is not None:
                    result_key = Mover360Dataset._build_predict_sample_id(scene_id, sample_id, frame_id)
            if result_key is None:
                return None
            repeat_id = entry.get('repeat_id')
            if repeat_id is not None:
                return f"{result_key}_{repeat_id:06d}"
            return result_key

        scene_id = entry.get('scene_id')
        camera_id = entry.get('camera_id')
        object_id = entry.get('object_id')

        if function in {'add', 'remove'} and None not in (scene_id, camera_id, object_id, entry.get('frame_id')):
            return f"{scene_id}_{camera_id}_{object_id}_{entry['frame_id']}_{function}"
        if function == 'move' and None not in (scene_id, camera_id, object_id, entry.get('frame2_id')):
            return f"{scene_id}_{camera_id}_{object_id}_{entry['frame2_id']}_move"
        if function == 'move' and None not in (scene_id, object_id, entry.get('frame_id')):
            sample_id = entry.get('sample_id', object_id)
            repeat_id = entry.get('repeat_id')
            if repeat_id is not None:
                return f"{Mover360Dataset._build_predict_sample_id(scene_id, sample_id, entry['frame_id'])}_{repeat_id:06d}"
            return Mover360Dataset._build_predict_sample_id(scene_id, sample_id, entry['frame_id'])
        return None

    def _count_result_key_matches(self, entries, results_set):
        match_count = 0
        for entry in entries:
            result_key = self._build_result_key(entry)
            if result_key is None:
                continue
            if result_key in results_set or any(result_key in result_name for result_name in results_set):
                match_count += 1
        return match_count

    def _build_standard_split(self, mode):
        new_data = []
        move_split_file = 'train.npy' if mode == 'train' else 'test.npy'
        move_split_path = os.path.join(self.inpaint_data_dir, move_split_file)
        add_split_file = 'train_add.npy' if mode == 'train' else 'test_add.npy'
        add_split_path = os.path.join(self.inpaint_data_dir, add_split_file)

        if not os.path.exists(move_split_path):
            raise FileNotFoundError(f"Cannot find split file: {move_split_path}")
        if not os.path.exists(add_split_path):
            raise FileNotFoundError(f"Cannot find split file: {add_split_path}")

        move_data = np.load(move_split_path)
        add_data = np.load(add_split_path)

        for d in move_data:
            scene_id, _, _, camera_id, object_id = d.split('/')
            object_folder = os.path.join(
                self.inpaint_data_dir,
                scene_id,
                'Saved',
                'MovieRenders_Normal',
                camera_id,
                object_id,
            )
            sample_tag = self._format_sample_tag(scene_id, camera_id, object_id)
            self._require_existing_dir(object_folder, f"move object folder for {sample_tag}")
            png_files = [f for f in os.listdir(object_folder) if f.endswith('.png')]

            if mode == 'train':
                selected_frames = self._sample_frame_pair_with_min_gap(png_files, min_index_gap=2)
            else:
                selected_frames = self._select_val_frame_pair(png_files)

            if selected_frames is None:
                self._warn_and_fail(
                    f"Cannot select valid move frame pair for {sample_tag}. "
                    f"png_count={len(png_files)}, mode={mode}"
                )

            frame1_id = selected_frames[0].replace('.png', '')
            frame2_id = selected_frames[1].replace('.png', '')
            label1 = 'simple_' + frame1_id + '_relative.txt'
            label2 = 'simple_' + frame2_id + '_relative.txt'

            new_data.append({
                'data_type': 'mp3d',
                'scene_id': scene_id,
                'camera_id': camera_id,
                'object_id': object_id,
                'frame1_id': frame1_id,
                'frame2_id': frame2_id,
                'label1': label1,
                'label2': label2,
                'function': 'move',
            })

        for d in add_data:
            scene_id, _, _, camera_id, object_id = d.split('/')
            frame_id = self._select_add_remove_frame(scene_id, camera_id, object_id, mode)

            base_entry = {
                'data_type': 'mp3d',
                'scene_id': scene_id,
                'camera_id': camera_id,
                'object_id': object_id,
                'frame_id': frame_id,
            }
            new_data.append({**base_entry, 'function': 'add'})
            new_data.append({**base_entry, 'function': 'remove'})

        return new_data

    def _build_predict_split(self):
        flat_predict_data = self._build_flat_predict_split()
        if flat_predict_data:
            return flat_predict_data

        new_data = []
        print(f"Scanning {self.predict_data_dir}...")

        # Predict inputs may point either to a parent directory containing multiple
        # scene folders or directly to a single scene root.
        for scene_id, scene_path in self._iter_predict_scene_roots():
            normal_path = os.path.join(scene_path, 'MovieRenders_Normal')
            box_mask_root = os.path.join(scene_path, 'MovieRenders_ObjectMaskBox')
            full_mask_root = os.path.join(scene_path, 'MovieRenders_ObjectMask')

            if not os.path.isdir(normal_path):
                continue

            for sample_id in sorted(os.listdir(normal_path)):
                sample_normal_path = os.path.join(normal_path, sample_id)
                if not os.path.isdir(sample_normal_path):
                    continue

                before_path = os.path.join(sample_normal_path, 'before')
                instruction_file = os.path.join(sample_normal_path, 'Instruction.txt')

                if not os.path.exists(before_path) or (
                    self.use_txt_prompt and not os.path.exists(instruction_file)
                ):
                    continue

                before_images = [
                    f for f in os.listdir(before_path)
                    if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')
                ]

                if not before_images:
                    continue

                for before_img in before_images:
                    frame_id = os.path.splitext(before_img)[0]
                    sample_box_mask_path = os.path.join(box_mask_root, sample_id)
                    sample_full_mask_path = os.path.join(full_mask_root, sample_id)

                    box_before_path = os.path.join(sample_box_mask_path, 'before', before_img)
                    box_after_path = os.path.join(sample_box_mask_path, 'after', before_img)
                    full_before_path = os.path.join(sample_full_mask_path, 'before', before_img)
                    full_after_path = os.path.join(sample_full_mask_path, 'after', before_img)

                    guidance_mask_before_path = self._first_existing_path(box_before_path, full_before_path)
                    guidance_mask_after_path = self._first_existing_path(box_after_path, full_after_path)
                    full_mask_before_path = self._first_existing_path(full_before_path, box_before_path)
                    full_mask_after_path = self._first_existing_path(full_after_path, box_after_path)

                    after_path = os.path.join(sample_normal_path, 'after')
                    after_pano_path = None
                    if os.path.exists(after_path):
                        after_img_path = os.path.join(after_path, before_img)
                        if os.path.exists(after_img_path):
                            after_pano_path = after_img_path
                        else:
                            after_images = [
                                f for f in os.listdir(after_path)
                                if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')
                            ]
                            if after_images:
                                after_pano_path = os.path.join(after_path, after_images[0])

                    new_data.append({
                        'scene_id': scene_id,
                        'object_id': sample_id,
                        'sample_id': sample_id,
                        'frame_id': frame_id,
                        'before_img': before_img,
                        'instruction_file': instruction_file,
                        'pano_path': after_pano_path,
                        'before_pano_path': os.path.join(before_path, before_img),
                        'pano_mask_path1': guidance_mask_after_path,
                        'pano_mask_path2': guidance_mask_before_path,
                        'full_mask_path1': full_mask_after_path,
                        'full_mask_path2': full_mask_before_path,
                        'function': 'move',
                        'predict_style_sample': True,
                    })

        return new_data

    def _build_flat_predict_split(self):
        function = self._normalized_test_function()
        new_data = []
        saved_roots = list(self._iter_flat_predict_saved_roots())
        if not saved_roots:
            return new_data

        print(f"Scanning {self.predict_data_dir} as flat {function} predict/test data...")

        for scene_id, saved_root in saved_roots:
            normal_root = os.path.join(saved_root, 'MovieRenders_Normal')
            box_mask_root = os.path.join(saved_root, 'MovieRenders_ObjectMaskBox')
            full_mask_root = os.path.join(saved_root, 'MovieRenders_ObjectMask')

            for sample_id in sorted(os.listdir(normal_root)):
                sample_normal_path = os.path.join(normal_root, sample_id)
                if not os.path.isdir(sample_normal_path):
                    continue

                prompt_file = os.path.join(sample_normal_path, f'{function}.txt')
                if self.use_txt_prompt and not os.path.exists(prompt_file):
                    continue

                before_pano_path = self._first_image_in_folder(
                    os.path.join(sample_normal_path, 'before')
                )
                after_pano_path = self._first_image_in_folder(
                    os.path.join(sample_normal_path, 'after')
                )
                background_pano_path = os.path.join(sample_normal_path, 'background.png')
                reference_image_path = os.path.join(sample_normal_path, 'reference.png')
                reference_mask_path = os.path.join(sample_normal_path, 'reference_mask.png')

                sample_box_mask_path = os.path.join(box_mask_root, sample_id)
                sample_full_mask_path = os.path.join(full_mask_root, sample_id)

                before_mask_name = os.path.basename(before_pano_path) if before_pano_path else 'before.png'
                after_mask_name = os.path.basename(after_pano_path) if after_pano_path else 'after.png'
                box_before_path = os.path.join(sample_box_mask_path, 'before', before_mask_name)
                box_after_path = os.path.join(sample_box_mask_path, 'after', after_mask_name)
                full_before_path = os.path.join(sample_full_mask_path, 'before', before_mask_name)
                full_after_path = os.path.join(sample_full_mask_path, 'after', after_mask_name)

                guidance_mask_before_path = self._first_existing_path(box_before_path, full_before_path)
                guidance_mask_after_path = self._first_existing_path(box_after_path, full_after_path)
                full_mask_before_path = self._first_existing_path(full_before_path, box_before_path)
                full_mask_after_path = self._first_existing_path(full_after_path, box_after_path)

                if function == 'add':
                    required_paths = (after_pano_path, background_pano_path, reference_image_path, reference_mask_path)
                    if not all(path is not None and os.path.exists(path) for path in required_paths):
                        continue
                    pano_path = after_pano_path
                    remove_pano_path = background_pano_path
                    pano_mask_path1 = guidance_mask_after_path
                    pano_mask_path2 = None
                    full_mask_path1 = full_mask_after_path
                    full_mask_path2 = None
                elif function == 'remove':
                    if self.remove_from_before:
                        # Move-chain stage 1: remove the SOURCE object from the
                        # BEFORE pano (the default remove benchmark edits the
                        # AFTER pano at the target-object mask).
                        required_paths = (before_pano_path, background_pano_path, guidance_mask_before_path)
                        if not all(path is not None and os.path.exists(path) for path in required_paths):
                            continue
                        pano_path = before_pano_path
                        remove_pano_path = background_pano_path
                        pano_mask_path1 = None
                        pano_mask_path2 = guidance_mask_before_path
                        full_mask_path1 = None
                        full_mask_path2 = full_mask_before_path
                    else:
                        required_paths = (after_pano_path, background_pano_path)
                        if not all(path is not None and os.path.exists(path) for path in required_paths):
                            continue
                        pano_path = after_pano_path
                        remove_pano_path = background_pano_path
                        pano_mask_path1 = None
                        pano_mask_path2 = guidance_mask_after_path
                        full_mask_path1 = None
                        full_mask_path2 = full_mask_after_path
                else:
                    required_paths = (before_pano_path, after_pano_path)
                    if not all(path is not None and os.path.exists(path) for path in required_paths):
                        continue
                    pano_path = after_pano_path
                    remove_pano_path = before_pano_path
                    pano_mask_path1 = guidance_mask_after_path
                    pano_mask_path2 = guidance_mask_before_path
                    full_mask_path1 = full_mask_after_path
                    full_mask_path2 = full_mask_before_path

                new_data.append({
                    'scene_id': scene_id,
                    'object_id': sample_id,
                    'sample_id': sample_id,
                    'frame_id': function,
                    'instruction_file': prompt_file,
                    'pano_prompt_path': prompt_file,
                    'pano_id_base': f"{scene_id}_{sample_id}_{function}",
                    'pano_path': pano_path,
                    'before_pano_path': before_pano_path,
                    'remove_pano_path': remove_pano_path,
                    'background_pano_path1': background_pano_path,
                    'background_pano_path2': background_pano_path,
                    'pano_mask_path1': pano_mask_path1,
                    'pano_mask_path2': pano_mask_path2,
                    'full_mask_path1': full_mask_path1,
                    'full_mask_path2': full_mask_path2,
                    'reference_image_path': reference_image_path if function == 'add' else None,
                    'reference_mask_path': reference_mask_path if function == 'add' else None,
                    'function': function,
                    'predict_style_sample': True,
                    'flat_predict_sample': True,
                    'prompt_is_flux2_full': True,
                })

        return new_data

    @staticmethod
    def _is_prediction_image_candidate(filename):
        _, ext = os.path.splitext(filename)
        return ext.lower() in {'.png', '.jpg', '.jpeg', '.webp'}

    def _resolve_result_prediction_path(self, result_dir_name):
        result_scene_dir = os.path.join(self.result_dir, result_dir_name)
        if os.path.isdir(result_scene_dir):
            image_candidates = [
                os.path.join(result_scene_dir, filename)
                for filename in sorted(os.listdir(result_scene_dir))
                if self._is_prediction_image_candidate(filename)
            ]
            if image_candidates:
                image_candidates.sort(
                    key=lambda path: (
                        0 if os.path.splitext(os.path.basename(path))[0].startswith(result_dir_name) else 1,
                        1 if os.path.splitext(os.path.basename(path))[0].isdigit() else 0,
                        os.path.basename(path),
                    )
                )
                return image_candidates[0]
        return os.path.join(self.result_dir, result_dir_name, 'pano.png')

    @classmethod
    def _sample_frame_pair_with_min_gap(cls, png_files, min_index_gap=2):
        indexed_files = []
        for png_file in png_files:
            frame_idx = cls._extract_frame_index(png_file)
            if frame_idx is not None:
                indexed_files.append((png_file, frame_idx))

        valid_pairs = [
            (src_file, dst_file)
            for src_file, src_idx in indexed_files
            for dst_file, dst_idx in indexed_files
            if src_file != dst_file and abs(src_idx - dst_idx) >= min_index_gap
        ]
        if not valid_pairs:
            return None
        return random.choice(valid_pairs)

    @classmethod
    def _select_val_frame_pair(cls, png_files):
        if not png_files:
            return None

        def sort_key(filename):
            frame_idx = cls._extract_frame_index(filename)
            if frame_idx is None:
                return (1, filename)
            return (0, f"{frame_idx:08d}_{filename}")

        ordered_files = sorted(png_files, key=sort_key)
        if len(ordered_files) == 1:
            return ordered_files[0], ordered_files[0]
        return ordered_files[0], ordered_files[-1]

    @staticmethod
    def _list_png_stems(folder_path):
        if not os.path.isdir(folder_path):
            return []
        return sorted(
            os.path.splitext(filename)[0]
            for filename in os.listdir(folder_path)
            if filename.endswith('.png')
        )

    @staticmethod
    def _list_image_files(folder_path):
        if folder_path is None or not os.path.isdir(folder_path):
            return []
        image_exts = ('.png', '.jpg', '.jpeg')
        return sorted(
            os.path.join(folder_path, filename)
            for filename in os.listdir(folder_path)
            if filename.lower().endswith(image_exts)
        )

    @classmethod
    def _first_image_in_folder(cls, folder_path):
        image_files = cls._list_image_files(folder_path)
        if not image_files:
            return None
        return image_files[0]

    def _normalized_test_function(self):
        function = str(self.test_function).lower()
        if function not in self._PREDICT_FUNCTIONS:
            raise ValueError(
                "test_function must be one of {'add', 'remove', 'move'}, "
                f"got {self.test_function!r}"
            )
        return function

    def _filter_entries_by_test_function(self, entries):
        function = self._normalized_test_function()
        return [entry for entry in entries if entry.get('function') == function]

    @staticmethod
    def _interleave_entries_by_function(entries, order=('move', 'add', 'remove')):
        grouped = {function: [] for function in order}
        extras = []
        for entry in entries:
            function = entry.get('function')
            if function in grouped:
                grouped[function].append(entry)
            else:
                extras.append(entry)

        interleaved = []
        max_len = max((len(grouped[function]) for function in order), default=0)
        for idx in range(max_len):
            for function in order:
                if idx < len(grouped[function]):
                    interleaved.append(grouped[function][idx])

        interleaved.extend(extras)
        return interleaved

    @staticmethod
    def _strip_flux2_image_prompt_prefix(prompt_text):
        prompt_text = str(prompt_text or '').strip()
        image_prefixes = (
            "Image 1 is the panorama to edit.",
            "Image 2 is the object reference.",
            "Image 3 marks the target region.",
            "Image 2 marks the target region.",
            "Image 2 marks the object to remove.",
            "Image 2 marks the source and target regions.",
        )
        changed = True
        while changed:
            changed = False
            for prefix in image_prefixes:
                if prompt_text.startswith(prefix):
                    prompt_text = prompt_text[len(prefix):].strip()
                    changed = True
        return prompt_text

    def _populate_task_prompts(self, data, function, instruction_text, prompt_is_flux2_full=False):
        prompt_text = str(instruction_text or '').strip()
        task_instruction = (
            self._strip_flux2_image_prompt_prefix(prompt_text)
            if prompt_is_flux2_full
            else prompt_text
        )
        if not task_instruction:
            task_instruction = self.predefined_prompt_text(function)

        if function == 'add':
            data['pano_prompt'] = self.build_flux2_add_input_prompt(task_instruction)
            data['pano_prompt_without_img'] = self.build_flux2_add_text_only_prompt(task_instruction)
            data['pano_prompt_with_mask'] = self.build_flux2_add_mask_prompt(task_instruction)
            data['pano_prompt_with_ref'] = self.build_flux2_ref_prompt(task_instruction)
            data['pano_prompt_with_ref_and_mask'] = (
                prompt_text if prompt_is_flux2_full else self.build_flux2_add_ref_and_mask_prompt(task_instruction)
            )
        elif function == 'remove':
            data['pano_prompt'] = self.build_flux2_remove_input_prompt(task_instruction)
            data['pano_prompt_without_img'] = self.build_flux2_remove_text_only_prompt(task_instruction)
            data['pano_prompt_with_mask'] = (
                prompt_text if prompt_is_flux2_full else self.build_flux2_remove_mask_prompt(task_instruction)
            )
            data['pano_prompt_with_ref'] = data['pano_prompt']
            data['pano_prompt_with_ref_and_mask'] = data['pano_prompt_with_mask']
        elif function == 'move':
            data['pano_prompt'] = self.build_flux2_move_input_prompt(task_instruction)
            data['pano_prompt_without_img'] = self.build_flux2_move_text_only_prompt(task_instruction)
            data['pano_prompt_with_mask'] = (
                prompt_text if prompt_is_flux2_full else self.build_flux2_move_mask_prompt(task_instruction)
            )
            data['pano_prompt_with_ref'] = data['pano_prompt']
            data['pano_prompt_with_ref_and_mask'] = data['pano_prompt_with_mask']
        else:
            raise ValueError(f"Unsupported predict/test function: {function!r}")

        data['pano_simple_prompt'] = data['pano_prompt']
        data['pano_simple_prompt_without_img'] = data['pano_prompt_without_img']
        data['pano_simple_prompt_with_mask'] = data['pano_prompt_with_mask']
        data['pano_simple_prompt_with_ref'] = data['pano_prompt_with_ref']
        data['pano_simple_prompt_with_ref_and_mask'] = data['pano_prompt_with_ref_and_mask']

    @staticmethod
    def _format_sample_tag(scene_id: str, camera_id: str | None = None, object_id: str | None = None) -> str:
        parts = [scene_id]
        if camera_id is not None:
            parts.append(camera_id)
        if object_id is not None:
            parts.append(object_id)
        return '/'.join(parts)

    def _warn_and_fail(self, message: str):
        warning_message = f"[Mover360 data integrity] {message}"
        warnings.warn(warning_message, stacklevel=2)
        raise FileNotFoundError(warning_message)

    def _require_existing_dir(self, folder_path: str, description: str) -> str:
        if not os.path.isdir(folder_path):
            self._warn_and_fail(f"Missing {description}: {folder_path}")
        return folder_path

    def _require_existing_file(self, file_path: str, description: str) -> str:
        if not os.path.exists(file_path):
            self._warn_and_fail(f"Missing {description}: {file_path}")
        return file_path

    def _select_add_remove_frame(self, scene_id, camera_id, object_id, mode):
        object_folder = os.path.join(
            self.inpaint_data_dir,
            scene_id,
            'Saved',
            'MovieRenders_Normal',
            camera_id,
            object_id,
        )
        mask_folder = os.path.join(
            self.inpaint_data_dir,
            scene_id,
            'Saved',
            'MovieRenders_ObjectMaskBox',
            camera_id,
            object_id,
        )
        sample_tag = self._format_sample_tag(scene_id, camera_id, object_id)
        self._require_existing_dir(object_folder, f"object folder for {sample_tag}")
        self._require_existing_dir(mask_folder, f"box mask folder for {sample_tag}")

        object_frames = set(self._list_png_stems(object_folder))
        mask_frames = set(self._list_png_stems(mask_folder))
        # Mover360 stores a shared clean background in the `_bj/{camera_id}_0000.png` slot.
        self._build_background_frame_path(scene_id, camera_id, object_id, frame_id='0000')
        shared_frames = sorted(object_frames & mask_frames)
        if not shared_frames:
            self._warn_and_fail(
                f"No shared frames between object renders and box masks for {sample_tag}. "
                f"object_frames={len(object_frames)}, mask_frames={len(mask_frames)}"
            )
        if mode == 'train':
            return random.choice(shared_frames)
        return shared_frames[0]

    def _build_background_frame_path(
        self,
        scene_id: str,
        camera_id: str,
        object_id: str,
        frame_id: str | None,
    ) -> str:
        if frame_id is None:
            self._warn_and_fail(
                f"Background frame_id is None for {self._format_sample_tag(scene_id, camera_id, object_id)}"
            )
        background_object_id = object_id if object_id.endswith("_bj") else f"{object_id}_bj"
        background_folder = os.path.join(
            self.inpaint_data_dir,
            scene_id,
            "Saved",
            "MovieRenders_Normal",
            camera_id,
            background_object_id,
        )
        sample_tag = self._format_sample_tag(scene_id, camera_id, object_id)
        self._require_existing_dir(background_folder, f"background folder for {sample_tag}")
        # Prefer the shared background for every frame instead of per-frame matches.
        shared_background_path = os.path.join(background_folder, f"{camera_id}_0000.png")
        if os.path.exists(shared_background_path):
            return shared_background_path
        legacy_shared_background_path = os.path.join(background_folder, "0000.png")
        if os.path.exists(legacy_shared_background_path):
            return legacy_shared_background_path
        background_path = os.path.join(background_folder, f"{frame_id}.png")
        if os.path.exists(background_path):
            return background_path
        self._warn_and_fail(
            f"Missing shared/per-frame background for {sample_tag}, checked: "
            f"{shared_background_path}, {legacy_shared_background_path}, {background_path}"
        )

    def _build_object_mask_frame_path(
        self,
        scene_id: str,
        camera_id: str,
        object_id: str,
        frame_id: str | None,
        use_box_mask: bool = False,
    ) -> str:
        if frame_id is None:
            self._warn_and_fail(
                f"Mask frame_id is None for {self._format_sample_tag(scene_id, camera_id, object_id)}"
            )
        mask_folder = "MovieRenders_ObjectMaskBox" if use_box_mask else "MovieRenders_ObjectMask"
        mask_path = os.path.join(
            self.inpaint_data_dir,
            scene_id,
            "Saved",
            mask_folder,
            camera_id,
            object_id,
            f"{frame_id}.png",
        )
        mask_desc = "box mask" if use_box_mask else "object mask"
        return self._require_existing_file(
            mask_path,
            f"{mask_desc} for {self._format_sample_tag(scene_id, camera_id, object_id)} frame {frame_id}",
        )

    @staticmethod
    def _swap_predict_mask_root(mask_path: str | None, use_box_mask: bool = False) -> str | None:
        if mask_path is None:
            return None
        source_root = "MovieRenders_ObjectMask" if use_box_mask else "MovieRenders_ObjectMaskBox"
        target_root = "MovieRenders_ObjectMaskBox" if use_box_mask else "MovieRenders_ObjectMask"
        candidate_path = mask_path.replace(source_root, target_root, 1)
        if candidate_path == mask_path or not os.path.exists(candidate_path):
            return None
        return candidate_path

    @staticmethod
    def _first_existing_path(*candidates: str | None) -> str | None:
        for candidate in candidates:
            if candidate is not None and os.path.exists(candidate):
                return candidate
        return None

    @staticmethod
    def _flat_scene_id_from_saved_root(saved_root):
        normalized = os.path.normpath(saved_root)
        if os.path.basename(normalized) == 'Saved':
            return os.path.basename(os.path.dirname(normalized))
        return os.path.basename(normalized)

    @classmethod
    def _is_flat_predict_saved_root(cls, saved_root):
        normal_root = os.path.join(saved_root, 'MovieRenders_Normal')
        if not os.path.isdir(normal_root):
            return False

        for sample_id in sorted(os.listdir(normal_root)):
            sample_path = os.path.join(normal_root, sample_id)
            if not os.path.isdir(sample_path):
                continue
            if (
                os.path.exists(os.path.join(sample_path, 'background.png'))
                or os.path.exists(os.path.join(sample_path, 'reference.png'))
                or os.path.exists(os.path.join(sample_path, 'add.txt'))
                or os.path.exists(os.path.join(sample_path, 'remove.txt'))
                or os.path.exists(os.path.join(sample_path, 'move.txt'))
            ):
                return True
        return False

    def _iter_flat_predict_saved_roots(self):
        candidates = [
            self.predict_data_dir,
            os.path.join(self.predict_data_dir, 'Saved'),
        ]

        if os.path.isdir(self.predict_data_dir):
            for child_name in sorted(os.listdir(self.predict_data_dir)):
                child_path = os.path.join(self.predict_data_dir, child_name)
                if not os.path.isdir(child_path):
                    continue
                candidates.append(child_path)
                candidates.append(os.path.join(child_path, 'Saved'))

        seen = set()
        for saved_root in candidates:
            saved_root = os.path.abspath(saved_root)
            if saved_root in seen:
                continue
            seen.add(saved_root)
            if self._is_flat_predict_saved_root(saved_root):
                yield self._flat_scene_id_from_saved_root(saved_root), saved_root

    def _iter_predict_scene_roots(self):
        direct_normal_path = os.path.join(self.predict_data_dir, 'MovieRenders_Normal')
        if os.path.isdir(direct_normal_path):
            scene_id = os.path.basename(os.path.normpath(self.predict_data_dir))
            yield scene_id, self.predict_data_dir
            return

        for scene_id in sorted(os.listdir(self.predict_data_dir)):
            scene_path = os.path.join(self.predict_data_dir, scene_id)
            if not os.path.isdir(scene_path):
                continue
            if os.path.isdir(os.path.join(scene_path, 'MovieRenders_Normal')):
                yield scene_id, scene_path

    @staticmethod
    def _build_predict_sample_id(scene_id: str, sample_id: str, frame_id: str | None) -> str:
        base_parts = [scene_id, sample_id]
        if frame_id is not None and frame_id != sample_id:
            base_parts.append(frame_id)
        return '_'.join(base_parts)

    def load_split(self, mode):
        if mode in {'train', 'val'}:
            new_data = self._build_standard_split(mode)
        elif mode == 'test':
            if list(self._iter_flat_predict_saved_roots()):
                new_data = self._build_predict_split()
            else:
                new_data = self._build_standard_split(mode)
            if self._initial_result_dir is not None and os.path.isdir(self._initial_result_dir):
                results_set = set(self.scan_results(self._initial_result_dir))
                predict_data = self._build_predict_split()
                standard_match_count = self._count_result_key_matches(new_data, results_set)
                predict_match_count = self._count_result_key_matches(predict_data, results_set)
                if predict_match_count > standard_match_count:
                    print(
                        f"Using predict-style test split for offline evaluation: "
                        f"matched {predict_match_count} result folders "
                        f"(standard split matched {standard_match_count})."
                    )
                    new_data = predict_data
        elif mode == 'predict':
            new_data = self._build_predict_split()
        else:
            raise FileNotFoundError(f"Unsupported split mode: {mode}")

        if mode in {'test', 'predict'}:
            new_data = self._filter_entries_by_test_function(new_data)

        if mode in {'val', 'test'}:
            # Keep early validation batches mixed so add/remove/move are all visible
            # even when the trainer limits the number of val batches.
            new_data = self._interleave_entries_by_function(new_data)

        # Limit dataset size
        # if mode == 'train':
        #     print(f"[Debug] Original training samples: {len(new_data)}")
        #     new_data = new_data[:3]  # Only use the first 2 samples for training
        #     print(f"[Debug] Using only first 2 samples for training")
        # elif mode in ['val', 'test']:
        #     print(f"[Debug] Original {mode} samples: {len(new_data)}")
        #     print(f"[Debug] Keeping all {mode} samples with interleaved functions")
        
        return new_data #9820 perspective views

    def scan_results(self, result_dir):
        # Scan all subdirectories in result_dir
        results = glob(os.path.join(result_dir, '*/'))
        parsed_results = []
        for r in results:
            # Use the directory name directly as the identifier, no special parsing
            dir_name = r.split('/')[-2]
            parsed_results.append(dir_name)
        return parsed_results

    def get_data(self, idx):
        data = self.data[idx].copy()
        
        # Handle predict mode (new format)
        if self.mode == 'predict' or data.get('predict_style_sample', False):
            scene_id = data['scene_id']
            object_id = data['object_id']
            frame_id = data.get('frame_id')
            sample_id = data.get('sample_id', object_id)
            function = str(data.get('function', self._normalized_test_function())).lower()
            
            predict_sample_id = data.get(
                'pano_id_base',
                self._build_predict_sample_id(scene_id, sample_id, frame_id),
            )
            if self.config['repeat_predict'] > 1:
                data['pano_id'] = f"{predict_sample_id}_{data['repeat_id']:06d}"
            else:
                data['pano_id'] = predict_sample_id
            
            instruction_file = data.get('instruction_file') or data.get('pano_prompt_path')
            should_read_prompt = self.use_txt_prompt or bool(data.get('prompt_is_flux2_full', False))
            if should_read_prompt and instruction_file is not None and os.path.exists(instruction_file):
                with open(instruction_file, 'r', encoding='utf-8') as f:
                    instruction_text = f.read().strip()
            else:
                instruction_text = self.predefined_prompt_text(function)
            
            data['instruction_text'] = instruction_text
            data['pano_prompt_path'] = instruction_file
            self._populate_task_prompts(
                data,
                function,
                instruction_text,
                prompt_is_flux2_full=bool(data.get('prompt_is_flux2_full', False)),
            )
            
            data['prompt'] = [''] * 10
            if function == 'add':
                data['ref_img_path'] = data.get('reference_image_path') or ''
                data['ref_mask_path'] = data.get('reference_mask_path') or ''
                data['has_ref'] = bool(data['ref_img_path'] and data['ref_mask_path'])
            else:
                data['ref_img_path'] = ''
                data['ref_mask_path'] = ''
                data['has_ref'] = False
            data['remove_pano_path'] = data.get('remove_pano_path', data.get('before_pano_path', None))
            if self.result_dir is not None:
                result_dir_name = data.get('result_dir_name', data['pano_id'])
                data['pano_pred_path'] = self._resolve_result_prediction_path(result_dir_name)
            
            return data

        scene_id = data['scene_id']
        camera_id = data['camera_id']
        object_id = data['object_id']
        data['prompt'] = [''] * 10

        if data['function'] in {'add', 'remove'}:
            frame_id = data['frame_id']
            object_image_path = os.path.join(
                self.data_dir,
                scene_id,
                'Saved',
                'MovieRenders_Normal',
                camera_id,
                object_id,
                f'{frame_id}.png',
            )
            object_image_path = self._require_existing_file(
                object_image_path,
                f"object image for {self._format_sample_tag(scene_id, camera_id, object_id)} frame {frame_id}",
            )
            background_image_path = self._build_background_frame_path(
                scene_id, camera_id, object_id, frame_id
            )
            mask_path = self._build_object_mask_frame_path(
                scene_id,
                camera_id,
                object_id,
                frame_id,
                use_box_mask=True,
            )
            full_mask_path = self._build_object_mask_frame_path(
                scene_id,
                camera_id,
                object_id,
                frame_id,
                use_box_mask=False,
            )

            data['pano_id'] = f"{scene_id}_{camera_id}_{object_id}_{frame_id}_{data['function']}"
            data['background_pano_path1'] = background_image_path
            data['background_pano_path2'] = background_image_path

            if data['function'] == 'add':
                instruction_text = 'add the object'
                data['pano_path'] = object_image_path
                data['remove_pano_path'] = background_image_path
                data['pano_mask_path1'] = mask_path
                data['pano_mask_path2'] = None
                data['full_mask_path1'] = full_mask_path
                data['full_mask_path2'] = None
                data['pano_prompt'] = self.build_flux2_add_input_prompt(instruction_text)
                data['pano_prompt_without_img'] = self.build_flux2_add_text_only_prompt(instruction_text)
                data['pano_prompt_with_mask'] = self.build_flux2_add_mask_prompt(instruction_text)
                data['pano_prompt_with_ref'] = self.build_flux2_ref_prompt(instruction_text)
                data['pano_prompt_with_ref_and_mask'] = self.build_flux2_add_ref_and_mask_prompt(instruction_text)
                data['pano_simple_prompt'] = data['pano_prompt']
                data['pano_simple_prompt_without_img'] = data['pano_prompt_without_img']
                data['pano_simple_prompt_with_mask'] = data['pano_prompt_with_mask']
                data['pano_simple_prompt_with_ref'] = data['pano_prompt_with_ref']
                data['pano_simple_prompt_with_ref_and_mask'] = data['pano_prompt_with_ref_and_mask']
            else:
                instruction_text = 'remove the object'
                data['pano_path'] = object_image_path
                data['remove_pano_path'] = background_image_path
                data['pano_mask_path1'] = None
                data['pano_mask_path2'] = mask_path
                data['full_mask_path1'] = None
                data['full_mask_path2'] = full_mask_path
                data['pano_prompt'] = self.build_flux2_remove_input_prompt(instruction_text)
                data['pano_prompt_without_img'] = self.build_flux2_remove_text_only_prompt(instruction_text)
                data['pano_prompt_with_mask'] = self.build_flux2_remove_mask_prompt(instruction_text)
                data['pano_prompt_with_ref'] = data['pano_prompt']
                data['pano_prompt_with_ref_and_mask'] = data['pano_prompt_with_mask']
                data['pano_simple_prompt'] = data['pano_prompt']
                data['pano_simple_prompt_without_img'] = data['pano_prompt_without_img']
                data['pano_simple_prompt_with_mask'] = data['pano_prompt_with_mask']
                data['pano_simple_prompt_with_ref'] = data['pano_prompt_with_ref']
                data['pano_simple_prompt_with_ref_and_mask'] = data['pano_prompt_with_ref_and_mask']

            return data

        frame1_id = data['frame1_id']
        frame2_id = data['frame2_id']
        label1 = data['label1']
        label = label1

        if 'absolute' in label:
            data['rotation_type'] = 'absolute'
        elif 'relative' in label:
            data['rotation_type'] = 'relative'

        data['pano_id'] = f"{scene_id}_{camera_id}_{object_id}_{frame2_id}_move"
        data['pano_path'] = os.path.join(
            self.data_dir,
            scene_id,
            'Saved',
            'MovieRenders_Normal',
            camera_id,
            object_id,
            f"{frame1_id}.png",
        )
        data['pano_path'] = self._require_existing_file(
            data['pano_path'],
            f"move source pano for {self._format_sample_tag(scene_id, camera_id, object_id)} frame {frame1_id}",
        )

        remove_pano_path = os.path.join(
            self.inpaint_data_dir,
            scene_id,
            'Saved',
            'MovieRenders_Normal',
            camera_id,
            object_id,
            f'{frame2_id}.png',
        )
        data['remove_pano_path'] = self._require_existing_file(
            remove_pano_path,
            f"move target pano for {self._format_sample_tag(scene_id, camera_id, object_id)} frame {frame2_id}",
        )
        data['background_pano_path1'] = self._build_background_frame_path(
            scene_id, camera_id, object_id, frame1_id
        )
        data['background_pano_path2'] = self._build_background_frame_path(
            scene_id, camera_id, object_id, frame2_id
        )
        data['pano_simple_prompt_path'] = os.path.join(
            self.inpaint_data_dir,
            scene_id,
            'Saved',
            'MovieRenders_Normal',
            camera_id,
            object_id,
            label1,
        )
        data['pano_mask_path1'] = self._build_object_mask_frame_path(
            scene_id,
            camera_id,
            object_id,
            frame1_id,
            use_box_mask=True,
        )
        data['pano_mask_path2'] = self._build_object_mask_frame_path(
            scene_id,
            camera_id,
            object_id,
            frame2_id,
            use_box_mask=True,
        )
        data['full_mask_path1'] = self._build_object_mask_frame_path(
            scene_id,
            camera_id,
            object_id,
            frame1_id,
            use_box_mask=False,
        )
        data['full_mask_path2'] = self._build_object_mask_frame_path(
            scene_id,
            camera_id,
            object_id,
            frame2_id,
            use_box_mask=False,
        )
        data['pano_prompt_path'] = os.path.join(
            self.inpaint_data_dir,
            scene_id,
            'Saved',
            'MovieRenders_Normal',
            camera_id,
            object_id,
            label1,
        )
        data['json_path'] = os.path.join(
            self.inpaint_data_dir,
            scene_id,
            'Saved',
            'MovieRenders_Normal',
            camera_id,
            object_id,
            'camera_pose.json',
        )

        return data




class Mover360_Base(PanoDataModule):
    def __init__(
            self,
            data_dir: str = 'data/UE5_data',
            inpaint_data_dir: str = 'data/UE5_data',
            s3d_data_dir: str = None,
            s3d_inpaint_data_dir: str = None,
            predict_data_dir: str = 'data/UE5_data/Test_all_new',

            *args,
            **kwargs
            ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.dataset_cls = Mover360Dataset
