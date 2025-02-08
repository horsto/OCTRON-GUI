
import os 
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sam2.build_sam import _load_checkpoint

def build_sam2_octron(
    config_file,
    ckpt_path=None,
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=False,
    **kwargs,
):
    '''
    Build the SAM2 model from config and checkpoint files.
    
    This is basically the original build_sam function.
    I am getting rid of the vos_optimized flag because I don't need it.
    
    Parameters
    ----------
    config_file : str
        Path to the SAM2 model config file (.yaml)
    ckpt_path : str
        Path to the SAM2 model checkpoint file (.pth)
    mode : str
        Mode to run the model in. Default is "eval"
    hydra_overrides_extra : list
        List of additional hydra overrides to apply. Default is []
    apply_postprocessing : bool
        Apply postprocessing to the model. Default is False. TODO: UNTESTED!
        That would be:
            dynamic_multimask_via_stability = True 
            dynamic_multimask_stability_delta = 0.05
            dynamic_multimask_stability_thresh = 0.98
            binarize_mask_from_pts_for_mem_enc = True
            fill_hole_area = 8
    **kwargs : dict
    '''
    
    # Find out which device to use
    # Will automatically select CUDA if available and MPS (on Mac)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "Support for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion.\n\n"
        )

    # Hydra configuration 
    hydra_overrides = [
        "++model._target_=octron.sam2_octron.helpers.sam2_octron.SAM2_octron",
    ]

    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)

    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model, device
