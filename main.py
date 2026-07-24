import os
import torch
import wandb
from models import *
from dataset import *
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from jsonargparse import lazy_instance
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.trainer import Trainer


class WandbGlobalStepCallback(Callback):
    @rank_zero_only
    def setup(self, trainer, pl_module, stage=None):
        logger = getattr(trainer, "logger", None)
        experiment = getattr(logger, "experiment", None)
        define_metric = getattr(experiment, "define_metric", None)
        if define_metric is None:
            return

        define_metric("trainer/global_step")
        define_metric("train/*", step_metric="trainer/global_step", step_sync=True)
        define_metric("val/*", step_metric="trainer/global_step", step_sync=True)


def cli_main():
    # remove slurm env vars due to this issue:
    if 'SLURM_NTASKS' in os.environ:
        del os.environ["SLURM_NTASKS"]
    if 'SLURM_JOB_NAME' in os.environ:
        del os.environ["SLURM_JOB_NAME"]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision('medium')

    wandb_id = os.environ.get('WANDB_RUN_ID')
    if wandb_id is None:
        wandb_id = wandb.util.generate_id()
        os.environ['WANDB_RUN_ID'] = wandb_id
    exp_dir = os.path.join('logs', wandb_id)
    os.makedirs(exp_dir, exist_ok=True)
    wandb_logger = lazy_instance(
        WandbLogger,
        project='Mover360',
        id=wandb_id,
        save_dir=exp_dir
        )

    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    class MyLightningCLI(LightningCLI):
        @staticmethod
        def _optional_config_value(config, key, default=None):
            if config is None:
                return default
            if hasattr(config, key):
                return getattr(config, key)
            get = getattr(config, "get", None)
            if get is not None:
                return get(key, default)
            return default

        def _subcommand_or_global_value(self, subcommand_config, key):
            value = self._optional_config_value(subcommand_config, key)
            if value is not None:
                return value
            return self._optional_config_value(self.config, key)

        def before_instantiate_classes(self):
            # Handle --ckpt parameter (loads model weights only)
            subcommand = self.config.subcommand
            subcommand_config = self.config[subcommand]
            ckpt_path = self._subcommand_or_global_value(subcommand_config, 'ckpt')
            if ckpt_path is not None:
                print(f"DEBUG: Found ckpt in config: {ckpt_path}")
                # Override ckpt_path with the command-line --ckpt value (could be None or a path)
                subcommand_config.model.init_args.ckpt_path = ckpt_path
                print(f"DEBUG: Set model.init_args.ckpt_path to: {subcommand_config.model.init_args.ckpt_path}")
            else:
                print("DEBUG: ckpt not found in config")

            for object_edit_ckpt_key in ('translate_ckpt', 'remove_ckpt'):
                object_edit_ckpt = self._subcommand_or_global_value(subcommand_config, object_edit_ckpt_key)
                if object_edit_ckpt is not None:
                    setattr(subcommand_config.model.init_args, object_edit_ckpt_key, object_edit_ckpt)

            for object_edit_size_key in ('object_edit_height', 'object_edit_width'):
                object_edit_size = self._subcommand_or_global_value(subcommand_config, object_edit_size_key)
                if object_edit_size is not None:
                    setattr(subcommand_config.model.init_args, object_edit_size_key, object_edit_size)

            if (
                hasattr(subcommand_config, 'model')
                and hasattr(subcommand_config.model, 'init_args')
                and hasattr(subcommand_config.model.init_args, 'residual_loss_max_t')
            ):
                alias_value = subcommand_config.model.init_args.residual_loss_max_t
                has_canonical = hasattr(subcommand_config.model.init_args, 'residual_loss_max_timestep')
                canonical_value = (
                    subcommand_config.model.init_args.residual_loss_max_timestep
                    if has_canonical else None
                )
                if canonical_value is None and alias_value is not None:
                    subcommand_config.model.init_args.residual_loss_max_timestep = alias_value
                del subcommand_config.model.init_args.residual_loss_max_t
            
            # Handle --resume parameter (restores full training state including optimizer, scheduler, epoch, etc.)
            resume_path = self._subcommand_or_global_value(subcommand_config, 'resume')
            if resume_path is not None:
                print(f"DEBUG: Resuming training from: {resume_path}")
                # Set ckpt_path for trainer to restore full training state
                subcommand_config.ckpt_path = resume_path
            
            # Handle --result_dir parameter
            result_dir = self._subcommand_or_global_value(subcommand_config, 'result_dir')
            if result_dir is not None:
                subcommand_config.data.init_args.result_dir = result_dir

            # predict_data_dir = self._subcommand_or_global_value(subcommand_config, 'predict_data_dir')
            # if predict_data_dir is not None:
            #     subcommand_config.data.init_args.predict_data_dir = predict_data_dir
            #     print(f"DEBUG: Set data.init_args.predict_data_dir to: {predict_data_dir}")
            # elif (
            #     hasattr(subcommand_config, 'data')
            #     and hasattr(subcommand_config.data, 'init_args')
            #     and hasattr(subcommand_config.data.init_args, 'predict_data_dir')
            # ):
            #     print(f"DEBUG: Using data.init_args.predict_data_dir: {subcommand_config.data.init_args.predict_data_dir}")

            test_function = self._subcommand_or_global_value(subcommand_config, 'test_function')
            if test_function is not None:
                subcommand_config.data.init_args.test_function = test_function

            if (
                hasattr(subcommand_config, 'data')
                and hasattr(subcommand_config.data, 'init_args')
                and hasattr(subcommand_config.data.init_args, 'dino_ref_fallback_to_external')
            ):
                subcommand_config.data.init_args.dino_ref_fallback_to_external = True
            
            # set result_dir, data and pano_height for evaluation
            if self.config.get('test', {}).get('model', {}).get('class_path') == 'models.EvalPanoGen':
                if self.config.test.data.init_args.result_dir is None:
                    result_dir = os.path.join(exp_dir, 'test')
                    self.config.test.data.init_args.result_dir = result_dir
                self.config.test.model.init_args.data = self.config.test.data.class_path.split('.')[-1]
                self.config.test.model.init_args.pano_height = self.config.test.data.init_args.pano_height
                self.config.test.data.init_args.val_batch_size = 1

        def add_arguments_to_parser(self, parser):
            parser.link_arguments("model.init_args.cam_sampler", "data.init_args.cam_sampler")
            # Add --ckpt shorthand for checkpoint path (only loads model weights)
            parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint file (loads model weights only)")
            # Add --resume shorthand for resuming training (restores full training state)
            parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming training (restores optimizer, scheduler, epoch, etc.)")
            # Add --result_dir shorthand for result directory
            parser.add_argument("--result_dir", type=str, default=None, help="Path to result directory")
            parser.add_argument("--predict_data_dir", type=str, default=None, help="Path to predict/test data directory")
            parser.add_argument("--test_function", type=str, default=None, help="Predict/test task: add, remove, or move")
            parser.add_argument("--translate_ckpt", type=str, default=None, help="3DiT/object-edit translate checkpoint for move")
            parser.add_argument("--remove_ckpt", type=str, default=None, help="3DiT/object-edit remove checkpoint for remove")
            parser.add_argument("--object_edit_height", type=int, default=None, help="3DiT/object-edit internal input height, divisible by 8")
            parser.add_argument("--object_edit_width", type=int, default=None, help="3DiT/object-edit internal input width, divisible by 8")

    cli = MyLightningCLI(
        trainer_class=Trainer,
        save_config_kwargs={'overwrite': True},
        parser_kwargs={'parser_mode': 'omegaconf', 'default_env': True},
        seed_everything_default=os.environ.get("LOCAL_RANK", 0),
        trainer_defaults={
            'strategy': 'ddp' if torch.cuda.device_count() > 1 else 'auto',
            'devices': 'auto',
            'log_every_n_steps': 10,
            'num_sanity_val_steps': 0,
            'limit_val_batches': 4,
            'benchmark': True,
            # 'max_steps': 30000,  # Set maximum iteration count, replacing max_epochs
            'max_epochs': 10,   # Comment out epoch-based settings
            'accumulate_grad_batches': 2,  # Gradient accumulation: accumulate gradients every 1 batch before updating parameters
            'precision': 'bf16-mixed', #'bf16-mixed',
            'callbacks': [lr_monitor, checkpoint_callback, WandbGlobalStepCallback()],#
            'logger': wandb_logger,
            # 'gradient_clip_val': 1.0,  # New: set gradient clipping value
            # 'gradient_clip_algorithm': 'norm'  # New: set gradient clipping algorithm
        })


if __name__ == '__main__':
    cli_main()
