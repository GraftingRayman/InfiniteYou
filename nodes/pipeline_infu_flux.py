import os
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers.models import FluxControlNetModel
from facexlib.recognition import init_recognition_model
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from .pipeline_flux_infusenet import FluxInfuseNetPipeline
from .resampler import Resampler
import comfy.model_management

class InfUFluxPipeline:
    def __init__(
        self,
        base_model_path: str,
        infiniteyou_root: str,
        model_version: str = "aes_stage2",
        image_proj_num_tokens: int = 8,
    ):
        # Validate inputs
        if model_version not in ["aes_stage2", "sim_stage1"]:
            raise ValueError(f"Invalid model_version: {model_version}")

        # Set device and paths
        self.device = comfy.model_management.get_torch_device()
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        
        # Construct model paths
        self.version_dir = os.path.join(
            infiniteyou_root,
            "infu_flux_v1.0",
            model_version
        )
        
        # Verify critical paths
        self._verify_paths()
        
        # Load models
        self.infusenet = FluxControlNetModel.from_pretrained(
            os.path.join(self.version_dir, "InfuseNetModel"),
            torch_dtype=self.dtype
        ).to(self.device)
        
        self.pipe = FluxInfuseNetPipeline.from_single_file(
            base_model_path,
            controlnet=self.infusenet,
            torch_dtype=self.dtype
        ).to(self.device)

        self.image_proj_model = self._load_projection_model(image_proj_num_tokens)
        self._init_face_models(infiniteyou_root)

    def _verify_paths(self):
        """Validate critical file structure"""
        required_paths = [
            os.path.join(self.version_dir, "InfuseNetModel"),
            os.path.join(self.version_dir, "image_proj_model.safetensors"),
            os.path.join(self.version_dir, "..", "..", "supports", "insightface")
        ]
        
        for path in required_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing required path: {path}\n"
                    "Directory structure should be:\n"
                    "models/checkpoints/infiniteyou/\n"
                    "├── infu_flux_v1.0/\n"
                    "│   ├── aes_stage2/\n"
                    "│   │   ├── InfuseNetModel/\n"
                    "│   │   └── image_proj_model.safetensors\n"
                    "└── supports/insightface/"
                )

    def _load_projection_model(self, num_tokens: int):
        """Load image projection model"""
        model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=512,
            output_dim=4096,
            ff_mult=4,
        )
        
        proj_path = os.path.join(self.version_dir, "image_proj_model.safetensors")
        model.load_state_dict(torch.load(proj_path, map_location="cpu"))
        return model.to(self.device).eval()

    def _init_face_models(self, root_path: str):
        """Initialize face analysis models"""
        insightface_path = os.path.join(root_path, "supports", "insightface")
        provider = "CUDAExecutionProvider" if "cuda" in str(self.device) else "CPUExecutionProvider"
        
        self.app_640 = FaceAnalysis(
            name="antelopev2",
            root=insightface_path,
            providers=[provider]
        ).prepare(ctx_id=0, det_size=(640, 640))

        self.arcface_model = init_recognition_model(
            "arcface", 
            device=self.device
        ).to(self.dtype).eval()

    # ... (keep other methods from previous implementation) ...