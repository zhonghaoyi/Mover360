import os
import warnings
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel, Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2ImageProcessor
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from einops import rearrange
from utils.pano import pad_pano, unpad_pano
from ..modules.utils import WandbLightningModule
from typing import Optional
from .processor_flux import FluxMultiModalProcessor


class PanoBase(WandbLightningModule):
    def __init__(
            self,
            pano_prompt_prefix: str = '',
            mv_pano_prompt: bool = False,
            ):
        super().__init__()
        self.save_hyperparameters()

    def add_pano_prompt_prefix(self, pano_prompt):
        if isinstance(pano_prompt, str):
            if pano_prompt == '':
                return ''
            if self.hparams.pano_prompt_prefix == '':
                return pano_prompt
            return ' '.join([self.hparams.pano_prompt_prefix, pano_prompt])
        return [self.add_pano_prompt_prefix(p) for p in pano_prompt]

    # def get_pano_prompt(self, batch):
    #     pano_prompt = batch['pano_prompt']
    #     return pano_prompt
    def get_pano_prompt(self, batch):
        if self.hparams.mv_pano_prompt:
            prompts = list(map(list, zip(*batch['prompt'])))
            pano_prompt = ['. '.join(p1) if p2 else '' for p1, p2 in zip(prompts, batch['pano_prompt'])]
        else:
            pano_prompt = batch['pano_prompt']
        return self.add_pano_prompt_prefix(pano_prompt)
    def get_forward_prompt(self, batch):
        forward_prompt = batch['forward_prompt']
        return forward_prompt
    def get_reverse_prompt(self, batch):
        reverse_prompt = batch['reverse_prompt']
        return reverse_prompt
    def get_edited_prompt(self, batch):
        edited_prompt = batch['edited_prompt']
        return edited_prompt

    def get_flux2_pano_prompts(
        self,
        batch,
        use_img_cfg=True,
        use_ref=False,
        use_mask_guidance=False,
        sample_has_ref=None,
    ):
        base_prompts = batch['pano_prompt']
        normalized_sample_has_ref = None
        if sample_has_ref is not None:
            normalized_sample_has_ref = []
            for has_ref in sample_has_ref:
                if isinstance(has_ref, torch.Tensor):
                    normalized_sample_has_ref.append(bool(has_ref.reshape(-1)[0].item()))
                else:
                    normalized_sample_has_ref.append(bool(has_ref))

        # Conditioning images are packed with a uniform slot layout across the batch.
        # When any sample uses reference conditioning, missing refs are zero-filled so
        # every sample still carries the same image numbering into the transformer.
        batch_uses_ref_slot = use_ref and (
            normalized_sample_has_ref is None or any(normalized_sample_has_ref)
        )

        if not use_img_cfg:
            prompts = batch.get('pano_prompt_without_img', base_prompts)
        elif batch_uses_ref_slot and use_mask_guidance:
            prompts = batch.get(
                'pano_prompt_with_ref_and_mask',
                batch.get('pano_prompt_with_mask', base_prompts),
            )
        elif use_mask_guidance:
            prompts = batch.get('pano_prompt_with_mask', base_prompts)
        elif batch_uses_ref_slot:
            prompts = batch.get('pano_prompt_with_ref', base_prompts)
        else:
            prompts = base_prompts

        # Respect dataset-level text dropout: an explicitly empty base prompt should stay empty
        # regardless of which Flux2 prompt variant is selected.
        if isinstance(base_prompts, str):
            if base_prompts == '':
                prompts = ''
        else:
            prompts = [
                '' if base_prompt == '' else prompt
                for base_prompt, prompt in zip(base_prompts, prompts)
            ]
        return self.add_pano_prompt_prefix(prompts)


class PanoGenerator(PanoBase):
    def __init__(
            self,
            lr: float = 2e-4,
            guidance_scale: float = 3,
            model_id: Optional[str] = 'black-forest-labs/FLUX.2-klein-base-4B',
            inference_timesteps: int = 50,  # Number of timesteps during inference
            image_use_prob: float = 0.99, # 0.01 for unconditioned input
            ref_use_prob: float = 1.0, # Reference image usage probability
            edit_mask_use_prob: float = 0.99, # Edit mask usage probability
            depth_use_prob: float = 0.9, # Depth guidance usage probability when mask guidance is enabled
            target_bbox_guidance_use_prob: float = 0.3, # Training probability of choosing target bbox guidance in exclusive bbox/point sampling
            point_guidance_use_prob: float = 0.7, # Training probability of choosing target point guidance in exclusive bbox/point sampling
            latent_pad: int = 8,
            pano_lora: bool = True,
            train_pano_lora: bool = True,
            lora_rank: int = 32,
            ckpt_path: Optional[str] = None,
            load_trainable_ckpt_only: bool = True,
            rot_diff: float = 90.0,
            unet_pad: bool = True,
            use_cube: bool = False,
            use_gradient_checkpointing: bool = False,
            compile_models: bool = True,
            use_mask_in_inference: bool = True,
            use_ref_in_inference: bool = True,
            inference_target_guidance_mode: str = "point", # `all` expands to point/bbox/mask variants during validation and saved predict/test outputs
            test_perspective_size: int = 512,
            test_perspective_context_scale: float = 2.0,
            test_perspective_min_fov: float = 35.0,
            test_perspective_max_fov: float = 178.0,
            test_perspective_mask_threshold: float = 0.0,
            clip_score_model_name: str = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
            dinov3_model_name: str = "facebook/dinov3-vith16plus-pretrain-lvd1689m",
            dinov3_cache_dir: Optional[str] = None,
            dinov3_local_files_only: bool = False,
            dinov3_hf_token: Optional[str] = None,
            dinov3_batch_size: int = 2, #每次分别编码 pred/gt 的透视图
            dinov3_device: Optional[str] = None,
            huggingface_cache: Optional[str] = None,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.trainable_params = []
        self.save_hyperparameters()
        if ckpt_path is not None:
            self.hparams.ckpt_path = ckpt_path
        if unet_pad is not True:
            warnings.warn(
                "`unet_pad` is currently kept for config compatibility but is not used by Mover360_depth/Flux2.",
                stacklevel=2,
            )
        if use_cube is not False:
            warnings.warn(
                "`use_cube` is currently kept for config compatibility but is not used by Mover360_depth/Flux2.",
                stacklevel=2,
            )
        self.load_shared()
        # self.instantiate_model()
        # if ckpt_path is not None:
        #     print(f"Loading weights from {ckpt_path}")
        #     state_dict = torch.load(ckpt_path, weights_only=True)['state_dict']
        #     self.convert_state_dict(state_dict)
        #     # try:
        #     self.load_state_dict(state_dict, strict=True)
            # except RuntimeError as e:
            #     print(e)
            #     self.load_state_dict(state_dict, strict=False)

    def exclude_eval_metrics(self, checkpoint):
        for key in list(checkpoint['state_dict'].keys()):
            if key.startswith('eval_metrics'):
                del checkpoint['state_dict'][key]

    def convert_state_dict(self, state_dict):
        current_state_keys = None
        try:
            current_state_keys = set(self.state_dict().keys())
        except Exception:
            current_state_keys = None

        def iter_key_variants(key):
            yield key

            stripped_key = key.replace("._orig_mod.", ".")
            if stripped_key != key:
                yield stripped_key

            if "._orig_mod." not in key:
                key_parts = key.split(".")
                for idx in range(1, len(key_parts)):
                    yield ".".join(key_parts[:idx] + ["_orig_mod"] + key_parts[idx:])

        normalized_state_dict = {}
        normalized_prefix_keys = 0
        for key, value in state_dict.items():
            normalized_key = key
            if current_state_keys is not None:
                for candidate_key in iter_key_variants(key):
                    if candidate_key in current_state_keys:
                        normalized_key = candidate_key
                        break
            else:
                normalized_key = key.replace("._orig_mod.", ".")

            if normalized_key != key:
                normalized_prefix_keys += 1

            if normalized_key not in normalized_state_dict:
                normalized_state_dict[normalized_key] = value

        if normalized_prefix_keys > 0:
            state_dict.clear()
            state_dict.update(normalized_state_dict)
            print(f"Normalized {normalized_prefix_keys} checkpoint keys against current model structure")

        lora_keys_count = 0
        for key in state_dict.keys():
            if 'mv_base_model.flux_transformer' in key and ('lora_A' in key or 'lora_B' in key):
                lora_keys_count += 1
        
        print(f"Detected {lora_keys_count} LoRA-related keys in weight file")

    def on_load_checkpoint(self, checkpoint):
        self.exclude_eval_metrics(checkpoint)
        self.convert_state_dict(checkpoint['state_dict'])

    def on_save_checkpoint(self, checkpoint):
        self.exclude_eval_metrics(checkpoint)

    def load_shared(self):
        flux_pipeline = Flux2KleinPipeline.from_pretrained(
            self.hparams.model_id, use_safetensors=True, cache_dir=self.hparams.huggingface_cache
        )
        self.tokenizer = flux_pipeline.tokenizer        # Qwen2TokenizerFast

        # Text encoder (Qwen3, frozen) — multi-layer hidden states
        self.text_encoder = flux_pipeline.text_encoder  # Qwen3ForCausalLM
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        # When True, inference() parks the text encoder on CPU after prompt encoding
        # (~8 GB freed before the denoising loop); encode_prompt moves it back on demand.
        # Must stay False for training: a CPU round-trip per step would dominate step time.
        self.offload_text_encoder_after_encode = False

        self.vae = flux_pipeline.vae  # AutoencoderKLFlux2 (latent_channels=32)
        self.vae.eval()
        self.vae.requires_grad_(False)
        if self.hparams.compile_models:
            self.vae = torch.compile(self.vae)

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) is not None else 8
        )
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.multimodal_processor = FluxMultiModalProcessor(self.tokenizer, max_image_size=1024)
        self.tokenizer_max_length = 512
        self.default_sample_size = 128
        self.text_encoder_out_layers = (9, 18, 27)
        self.is_distilled = getattr(flux_pipeline.config, "is_distilled", False)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.hparams.model_id,
            subfolder="scheduler",
            torch_dtype=torch.float32,
            use_safetensors=True,
            cache_dir=self.hparams.huggingface_cache
        )

        del flux_pipeline.transformer

    def add_lora(self, flux):
        from peft import LoraConfig, get_peft_model

        # Dynamically build target module list to handle Flux2's dual architecture:
        # Double blocks: to_q, to_k, to_v, to_out.0 (Linear inside ModuleList), add_q/k/v_proj, to_add_out
        # Single blocks: to_qkv_mlp_proj, to_out (plain Linear)
        attn_targets = {'to_q', 'to_k', 'to_v', 'to_out.0', 'add_q_proj', 'add_k_proj', 'add_v_proj', 
                        'to_add_out', 'to_qkv_mlp_proj', 'to_out'}
        target_modules = []
        for name, mod in flux.named_modules():
            if not isinstance(mod, torch.nn.Linear):
                continue
            short = name.split('.')[-1]
            if name.endswith('.0'):
                short = '.'.join(name.split('.')[-2:])
            if short in attn_targets:
                # Skip double block's to_out (ModuleList) — we use to_out.0 instead
                if name.endswith('to_out') and 'single_transformer' not in name:
                    continue
                target_modules.append(name)

        lora_config = LoraConfig(
            r=self.hparams.lora_rank,  
            lora_alpha=self.hparams.lora_rank,  
            target_modules=target_modules,
            lora_dropout=0., 
            bias="none",       
        )
        flux = get_peft_model(flux, lora_config)
        print("add LoRA adapter completed.")

        lora_trainable_params = [p for p in flux.parameters() if p.requires_grad]
        return (lora_trainable_params, 1.0)



    def load_branch(self, add_lora, train_lora):
        flux = Flux2Transformer2DModel.from_pretrained(
            self.hparams.model_id, subfolder="transformer", torch_dtype=torch.bfloat16,  # bf16 to save ~8GB GPU memory
            use_safetensors=True, cache_dir=self.hparams.huggingface_cache
        )
        self.transformer_channel = flux.config.in_channels
        if self.hparams.use_gradient_checkpointing:
            flux.enable_gradient_checkpointing()

        if hasattr(flux, "add_adapter"):
            print("model supports add_adapter method, using it to add LoRA...")

        if add_lora:
            params = self.add_lora(flux)
            if train_lora:
                self.trainable_params.append(params)

        if self.hparams.compile_models:
            flux = torch.compile(flux)

        return flux

    def load_pano(self):
        return self.load_branch(
            self.hparams.pano_lora,
            self.hparams.train_pano_lora,
        )

    @torch.no_grad()
    def encode_prompt(self, prompts, device, max_sequence_length=512):
        """
        Encode text using Qwen3 encoder (multi-layer hidden states concatenation).

        Flux2 Klein uses a single Qwen3ForCausalLM text encoder. Intermediate layer
        hidden states are stacked and reshaped to produce prompt embeddings with
        joint_attention_dim = num_layers * hidden_dim (e.g., 3 * 5120 = 15360).

        Args:
            prompts: list of text strings
            device: target device
            max_sequence_length: max token length for Qwen3 tokenizer

        Returns:
            prompt_embeds: [B, seq_len, joint_attention_dim]
            text_ids: [B, seq_len, 4]
        """
        dtype = self.text_encoder.dtype

        if isinstance(prompts, str):
            prompts = [prompts]

        all_input_ids = []
        all_attention_masks = []

        for single_prompt in prompts:
            messages = [{"role": "user", "content": single_prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )
            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0)
        attention_mask = torch.cat(all_attention_masks, dim=0)

        # Move text encoder to GPU temporarily for encoding, then back to CPU
        text_encoder_device = next(self.text_encoder.parameters()).device
        if text_encoder_device != device:
            self.text_encoder = self.text_encoder.to(device)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack(
            [output.hidden_states[k] for k in self.text_encoder_out_layers], dim=1
        )
        out = out.to(dtype=dtype, device=device)

        # Offloading back to CPU (if enabled) is done by the caller once ALL prompt
        # encodes for the request are finished, so positive+negative encodes share
        # a single CPU->GPU round-trip. See offload_text_encoder().
        torch.cuda.empty_cache()

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, num_channels * hidden_dim
        )

        text_ids = self._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)

        return prompt_embeds, text_ids

    def offload_text_encoder(self):
        """Park the frozen Qwen3 text encoder on CPU and release its GPU memory.

        encode_prompt() moves it back to the target device on demand, so this is
        safe to call between requests. No-op if it is already on CPU.
        """
        if next(self.text_encoder.parameters()).device.type == "cpu":
            return
        self.text_encoder = self.text_encoder.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _prepare_text_ids(x):
        """Generate 4D position IDs (T, H, W, L) for text tokens."""
        B, L, _ = x.shape
        out_ids = []
        for i in range(B):
            t = torch.arange(1)
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)
            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)
        return torch.stack(out_ids)

    @torch.no_grad()
    def encode_text(self, text):
        """Legacy text encoding (uses Qwen3 via encode_prompt)."""
        device = next(self.text_encoder.parameters()).device
        prompt_embeds, _ = self.encode_prompt(text, device)
        return prompt_embeds


    @torch.no_grad()
    def encode_image_latents(
        self,
        input_pixel_values,
        device: torch.device,
    ):
        """
        Encode images via Flux2 VAE without patchifying.

        Returns:
            list of latent tensors [1, 32, H/8, W/8] in the native VAE latent space.
        """
        dtype = self.vae.dtype

        input_img_latents = []
        for img in input_pixel_values:
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            latent = self.vae.encode(img.to(device, dtype)).latent_dist.mode()
            input_img_latents.append(latent)
        return input_img_latents

    @torch.no_grad()
    def encode_image(
        self,
        input_pixel_values,
        device: torch.device,
    ):
        """
        Encode images via Flux2 VAE: encode → patchify → batch norm normalize.

        Flux2 VAE has 32 latent channels. After encoding, latents are patchified
        (2x2 spatial → 4x channels) to get [B, 128, H/16, W/16], then normalized
        using the VAE's running batch norm statistics.

        Args:
            input_pixel_values: normalized pixel values of input images
            device: target device
        Returns:
            list of latent tensors [1, 128, H/16, W/16] (patchified + BN normalized)
        """
        dtype = self.vae.dtype

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype)
        latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(device, dtype)

        input_img_latents = self.encode_image_latents(input_pixel_values, device=device)

        patchified_latents = []
        for latent in input_img_latents:
            latent = self._patchify_latents(latent)
            latent = (latent - latents_bn_mean) / latents_bn_std
            patchified_latents.append(latent)
        return patchified_latents

    @staticmethod
    def _patchify_latents(latents):
        """Patchify: [B, C, H, W] → [B, C*4, H/2, W/2] (2x2 spatial → channel)."""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _unpatchify_latents(latents):
        """Unpatchify: [B, C*4, H/2, W/2] → [B, C, H, W] (channel → 2x2 spatial)."""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels // 4, 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels // 4, height * 2, width * 2)
        return latents

    def check_inputs(
        self,
        prompt,
        input_images,
        height,
        width,
        use_input_image_size_as_output,
        callback_on_step_end_tensor_inputs=None,
    ):
        if input_images is not None:
            if len(input_images) != len(prompt):
                raise ValueError(
                    f"The number of prompts: {len(prompt)} does not match the number of input images: {len(input_images)}."
                )
            # for i in range(len(input_images)):
            #     if input_images[i] is not None:
            #         if not all(f"<img><|image_{k + 1}|></img>" in prompt[i] for k in range(len(input_images[i]))):
            #             raise ValueError(
            #                 f"prompt `{prompt[i]}` doesn't have enough placeholders for the input images `{input_images[i]}`"
            #             )

        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            print(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if use_input_image_size_as_output:
            if input_images is None or input_images[0] is None:
                raise ValueError(
                    "`use_input_image_size_as_output` is set to True, but no input image was found. If you are performing a text-to-image task, please set it to False."
                )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
            
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()
        
    def pad_pano(self, pano, latent=False, patchified=False):
        padding = self.hparams.latent_pad
        if patchified:
            padding = padding // 2
        elif not latent:
            padding *= 8
        return pad_pano(pano, padding=padding)

    def unpad_pano(self, pano_pad, latent=False, patchified=False):
        padding = self.hparams.latent_pad
        if patchified:
            padding = padding // 2
        elif not latent:
            padding *= 8
        return unpad_pano(pano_pad, padding=padding)

    def gen_cls_free_guide_pair(self, *inputs):
        result = []
        for input in inputs:
            if input is None:
                result.append(None)
            elif isinstance(input, dict):
                result.append({k: torch.cat([v]*3) for k, v in input.items()})
            elif isinstance(input, list):
                result.append([torch.cat([v]*3) for v in input])
            else:
                result.append(torch.cat([input]*3))
        return result



    def rotate_latent(self, pano_latent, degree=None):
        if degree is None:
            degree = self.hparams.rot_diff
        if degree % 360 == 0:
            return pano_latent
        return torch.roll(pano_latent, int(degree / 360 * pano_latent.shape[-1]), dims=-1)

    @torch.no_grad()
    def decode_latent(self, latents, vae):
        """Decode patchified+BN latents: BN denormalize → unpatchify → VAE decode."""
        b = latents.shape[0]
        latents = rearrange(latents, 'b m c h w -> (b m) c h w')
        latents = latents.to(vae.dtype)

        latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = latents * latents_bn_std + latents_bn_mean

        latents = self._unpatchify_latents(latents)
        image = vae.decode(latents).sample
        image = rearrange(image, '(b m) c h w -> b m c h w', b=b)
        return image.to(self.dtype)

    def configure_optimizers(self):
        param_groups = []
        for params, lr_scale in self.trainable_params:
            # Filter parameters where requires_grad is True
            param_groups.append({"params": params, "lr": self.hparams.lr * lr_scale})
        
        if not param_groups:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Warning: No trainable parameters found for the optimizer.!!!!!!!!!!!!!!!!!!!!!!!!!")
            # return None 
            pass

        optimizer = torch.optim.AdamW(param_groups)

        # scheduler = {
        #     'scheduler': CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7),
        #     'interval': 'epoch',  # update the learning rate after each epoch
        #     'name': 'cosine_annealing_lr',
        # }
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=self.trainer.estimated_stepping_batches, eta_min=1e-7), # Use estimated_stepping_batches as T_max
            'interval': 'step', 
            'name': 'cosine_annealing_lr',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def _slice_value_for_saving(value, batch_size, index):
        if isinstance(value, torch.Tensor):
            return value[index:index+1]

        if isinstance(value, dict):
            return {
                key: PanoGenerator._slice_value_for_saving(item, batch_size, index)
                for key, item in value.items()
            }

        if isinstance(value, list):
            if not value:
                return []
            if len(value) == batch_size:
                return value[index:index+1]
            if all(isinstance(item, torch.Tensor) and item.shape[0] == batch_size for item in value):
                return [item[index:index+1] for item in value]
            if all(isinstance(item, (list, tuple)) and len(item) == batch_size for item in value):
                return [item[index:index+1] for item in value]
            return value

        if isinstance(value, tuple):
            if not value:
                return ()
            if len(value) == batch_size:
                return value[index:index+1]
            if all(isinstance(item, torch.Tensor) and item.shape[0] == batch_size for item in value):
                return tuple(item[index:index+1] for item in value)
            if all(isinstance(item, (list, tuple)) and len(item) == batch_size for item in value):
                return tuple(item[index:index+1] for item in value)
            return value

        return value

    @staticmethod
    def _slice_batch_for_saving(batch, index):
        batch_size = len(batch['function'])
        return {
            key: PanoGenerator._slice_value_for_saving(value, batch_size, index)
            for key, value in batch.items()
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        for sample_idx in range(len(batch['function'])):
            sample_batch = self._slice_batch_for_saving(batch, sample_idx)
            output_dir = os.path.join(self.logger.save_dir, 'test', sample_batch['pano_id'][0])
            self.inference_and_save(sample_batch, output_dir)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        for sample_idx in range(len(batch['function'])):
            sample_batch = self._slice_batch_for_saving(batch, sample_idx)
            output_dir = os.path.join(self.logger.save_dir, 'predict', sample_batch['pano_id'][0])
            self.inference_and_save(sample_batch, output_dir, 'jpg')
