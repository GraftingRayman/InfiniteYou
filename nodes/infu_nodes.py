import os
import sys
import torch
import numpy as np
from PIL import Image
import cv2
import math
import torch.nn as nn
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from facexlib.recognition import init_recognition_model
import folder_paths
from diffusers.models import FluxControlNetModel
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed
import bitsandbytes as bnb  # Add this line


# Add local InfiniteYou project to path to import its modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "InfiniteYou"))

# Import Resampler model
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Changed to 8-bit Linear layers
        self.to_q = bnb.nn.Linear8bitLt(dim, inner_dim, bias=False)
        self.to_kv = bnb.nn.Linear8bitLt(dim, inner_dim * 2, bias=False)
        self.to_out = bnb.nn.Linear8bitLt(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)

class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
    ):
        super().__init__()
        
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        
        # Changed to 8-bit projection
        self.proj_in = bnb.nn.Linear8bitLt(embedding_dim, dim)
        self.proj_out = bnb.nn.Linear8bitLt(dim, output_dim)  # Changed
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1)
        
        x = self.proj_in(x)
        
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
            
        latents = self.proj_out(latents)
        return self.norm_out(latents)

# Define utility functions from InfiniteYou
def resize_and_pad_image(source_img, target_img_size):
    # Get original and target sizes
    source_img_size = source_img.size
    target_width, target_height = target_img_size
    
    # Determine the new size based on the shorter side of target_img
    if target_width <= target_height:
        new_width = target_width
        new_height = int(target_width * (source_img_size[1] / source_img_size[0]))
    else:
        new_height = target_height
        new_width = int(target_height * (source_img_size[0] / source_img_size[1]))
    
    # Resize the source image using LANCZOS interpolation for high quality
    resized_source_img = source_img.resize((new_width, new_height), Image.LANCZOS)
    
    # Compute padding to center resized image
    pad_left = (target_width - new_width) // 2
    pad_top = (target_height - new_height) // 2
    
    # Create a new image with white background
    padded_img = Image.new("RGB", target_img_size, (255, 255, 255))
    padded_img.paste(resized_source_img, (pad_left, pad_top))
    
    return padded_img

def extract_arcface_embedding(in_image, landmark, arcface_model=None):
    kps = landmark
    arc_face_image = face_align.norm_crop(in_image, landmark=np.array(kps), image_size=112)
    arc_face_image = torch.from_numpy(arc_face_image).unsqueeze(0).permute(0,3,1,2) / 255.
    arc_face_image = 2 * arc_face_image - 1
    arc_face_image = arc_face_image.cuda().contiguous()
    if arcface_model is None:
        arcface_model = init_recognition_model('arcface', device='cuda')
    face_emb = arcface_model(arc_face_image)[0]  # [512], normalized
    return face_emb

def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


# Define ComfyUI node for loading InfiniteYou model
class LoadInfuModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "infu_flux_version": (["v1.0"], {"default": "v1.0"}),
                "model_version": (["aes_stage2", "sim_stage1"], {"default": "aes_stage2"}),
            },
        }

    RETURN_TYPES = ("INFU_MODEL",)
    RETURN_NAMES = ("infu_model",)
    FUNCTION = "load_model"
    CATEGORY = "InfiniteYou"

    def load_model(self, infu_flux_version, model_version):
        # 准备路径
        model_path = os.path.join(folder_paths.models_dir, "InfiniteYou")
        infu_model_path = os.path.join(model_path, f"infu_flux_{infu_flux_version}", model_version)
        infusenet_path = os.path.join(infu_model_path, 'InfuseNetModel')
        image_proj_model_path = os.path.join(infu_model_path, 'image_proj_model.bin')
        
        # 加载图像投影模型
        num_tokens = 8  # image_proj_num_tokens
        image_emb_dim = 512
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=4096,
            ff_mult=4,
        )
        
        # 加载权重
        ipm_state_dict = torch.load(image_proj_model_path, map_location="cpu")
        if 'image_proj' in ipm_state_dict:
            state_dict = ipm_state_dict['image_proj']
        else:
            state_dict = ipm_state_dict
            
        # 直接加载权重,不需要转换键值
        image_proj_model.load_state_dict(state_dict)
        image_proj_model.to('cuda')
        image_proj_model.eval()
        
        # 打包模型
        infu_model = {
            "image_proj_model": image_proj_model,
            "model_version": model_version,
            "model_path": infu_model_path,
            "infusenet_path": infusenet_path
        }
        
        print(f"InfiniteYou model loaded: {model_version}")
        return (infu_model,)


# 首先创建一个扩展的 FluxControlNetModel 类
class AdvancedFluxControlNetModel(FluxControlNetModel):
    _no_split_modules = ["Attention", "PerceiverAttention", "Resampler"]  # Add all critical modules

    def __init__(self, *args, **kwargs):
        # 基本的 Advanced-ControlNet 接口属性
        self.previous_controlnet = None
        self.control_hint = None
        self.strength = 1.0
        self.start_percent = 0.0
        self.end_percent = 1.0

        # 参考 pulidflux.py 的实现，添加必要的属性
        self.double_interval = 2
        self.single_interval = 4
        
        # 添加 flux 相关的属性
        self.guidance_scale = 1.0
        self.guidance_embeds = True
        self.controlnet_cond = None
        self.controlnet_mode = None
        self.txt_ids = None
        self.img_ids = None
        self.joint_attention_kwargs = None

        # 在调用父类初始化之前，确保配置中包含正确的时间投影层类型
        if 'config' in kwargs:
            config = kwargs['config']
            config.time_embedding_type = 'flux_guidance'  # 确保使用正确的时间嵌入类型
            config.guidance_embeds = True  # 启用引导嵌入
            
            # 创建正确的时间投影层
            embedding_dim = getattr(config, "embedding_dim", 1280)
            kwargs['time_proj'] = CombinedTimestepGuidanceTextProjEmbeddings(
                embedding_dim=embedding_dim,
                pooled_projection_dim=embedding_dim  # 设置为与 embedding_dim 相同
            )
        
        super().__init__(*args, **kwargs)

    # 添加 get_extra_hooks 方法
    def get_extra_hooks(self):
        return {
            "controlnet_cond": self.controlnet_cond,
            "controlnet_mode": self.controlnet_mode,
            "txt_ids": self.txt_ids,
            "img_ids": self.img_ids,
            "joint_attention_kwargs": self.joint_attention_kwargs,
        }

    @classmethod
    def from_pretrained(cls, pretrained_model_path, **kwargs):
        print(f"Loading model from {pretrained_model_path}")
        print(f"Original kwargs: {kwargs}")
        
        # 创建基本配置
        config_dict = {
            'time_embedding_type': 'flux_guidance',
            'guidance_embeds': True,
            'embedding_dim': 1280  # 默认值
        }
        
        # 创建时间投影层
        embedding_dim = config_dict['embedding_dim']
        time_proj = CombinedTimestepGuidanceTextProjEmbeddings(
            embedding_dim=embedding_dim,
            pooled_projection_dim=embedding_dim  # 设置为与 embedding_dim 相同
        )
        
        # # 从 kwargs 中移除可能存在的 config
        # if 'config' in kwargs:
        #     print("Removing config from kwargs")
        #     kwargs.pop('config')
        
        # # 准备新的 kwargs
        # new_kwargs = {
        #     'time_proj': time_proj,
        #     **kwargs
        # }
        # print(f"New kwargs: {new_kwargs}")
        
        try:
            # 使用父类的 from_pretrained 方法加载模型
            base_model = FluxControlNetModel.from_pretrained(
                pretrained_model_path,
                **kwargs
            )
            print("Base model loaded successfully")
            
            # 创建我们的高级模型实例
            model = cls()
            
            # 复制基础模型的状态
            model.load_state_dict(base_model.state_dict())
            
            # 设置配置
            model.config = base_model.config
            for key, value in config_dict.items():
                setattr(model.config, key, value)
            
            # 复制其他必要的属性
            model.dtype = base_model.dtype
            model.device = base_model.device
            model.time_proj = base_model.time_proj
            
            # 设置默认属性
            model.guidance_scale = 1.0
            model.guidance_embeds = True
            model.controlnet_cond = None
            model.controlnet_mode = None
            model.txt_ids = None
            model.img_ids = None
            model.joint_attention_kwargs = None
            
            print("Advanced model created successfully")
            return model
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print(f"Error type: {type(e)}")
            raise

    def set_cond_hint(self, control_hint, strength, start_percent_end_percent, vae=None):
        self.control_hint = control_hint
        self.strength = strength
        self.start_percent, self.end_percent = start_percent_end_percent
        return self

    def set_previous_controlnet(self, prev_cnet):
        self.previous_controlnet = prev_cnet
        return self

    def copy(self):
        new_model = AdvancedFluxControlNetModel(*self._args, **self._kwargs)
        new_model.load_state_dict(self.state_dict())
        
        # 复制基本属性
        new_model.previous_controlnet = self.previous_controlnet
        new_model.control_hint = self.control_hint
        new_model.strength = self.strength
        new_model.start_percent = self.start_percent
        new_model.end_percent = self.end_percent
        
        # 复制 flux 相关属性
        new_model.guidance_scale = self.guidance_scale
        new_model.guidance_embeds = self.guidance_embeds
        new_model.controlnet_cond = self.controlnet_cond
        new_model.controlnet_mode = self.controlnet_mode
        new_model.txt_ids = self.txt_ids
        new_model.img_ids = self.img_ids
        new_model.joint_attention_kwargs = self.joint_attention_kwargs
        
        return new_model

    def verify_all_weights(self):
        return True

    def disarm(self):
        return self

    def get_models(self):
        return [self]

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

# Define ComfyUI node for applying InfiniteYou model
class ApplyInfu:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "infu_model": ("INFU_MODEL", ),
                "face_analyzers": ("INFU_FACEANALYSIS", ),
                "id_image": ("IMAGE", ),
                "infusenet_conditioning_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "infusenet_guidance_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "infusenet_guidance_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "control_image": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "process"
    CATEGORY = "InfiniteYou"

    def process(self, model, positive, negative, infu_model, face_analyzers, id_image,
               infusenet_conditioning_scale, infusenet_guidance_start, 
               infusenet_guidance_end, control_image=None):
        # 提取模型组件
        image_proj_model = infu_model["image_proj_model"]
        infusenet_path = infu_model["infusenet_path"]
        
        # 提取人脸分析器
        app_640 = face_analyzers["app_640"]
        app_320 = face_analyzers["app_320"]
        app_160 = face_analyzers["app_160"]
        arcface_model = face_analyzers["arcface_model"]
        
        # 加载 InfuseNet ControlNet 模型
        infusenet = AdvancedFluxControlNetModel.from_pretrained(
            infusenet_path,
            load_in_8bit=True,
#            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16
        )
        if torch.cuda.is_available():
            infusenet = infusenet.to("cuda")
        else:
            infusenet = infusenet.to("cpu")
        
        # 将 controlnet 添加到 model 中
        model.model.controlnet = infusenet
        
        # 转换 ID 图像为 PIL 格式
        if isinstance(id_image, torch.Tensor):
            if id_image.dim() == 4 and id_image.shape[0] == 1:
                id_image = id_image.squeeze(0)
            
            id_image_np = (id_image.cpu().numpy() * 255).astype(np.uint8)
            id_image_pil = Image.fromarray(id_image_np)
        else:
            id_image_pil = id_image
        
        # 处理 ID 图像提取嵌入
        id_image_cv2 = cv2.cvtColor(np.array(id_image_pil), cv2.COLOR_RGB2BGR)
        
        # 使用不同尺寸检测人脸
        face_info = app_640.get(id_image_cv2)
        if len(face_info) == 0:
            face_info = app_320.get(id_image_cv2)
        if len(face_info) == 0:
            face_info = app_160.get(id_image_cv2)
        
        if len(face_info) == 0:
            raise ValueError('无法在输入的 ID 图像中检测到人脸')
            
        # 使用最大的人脸
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
        landmark = face_info['kps']
        
        # 提取人脸嵌入
        id_embed = extract_arcface_embedding(id_image_cv2, landmark, arcface_model)
        id_embed = id_embed.clone().unsqueeze(0).float().cuda()
        id_embed = id_embed.reshape([1, -1, 512])
        id_embed = id_embed.to(device='cuda', dtype=torch.bfloat16)
        
        # 通过图像投影模型处理
        with torch.no_grad():
            id_embed = image_proj_model(id_embed)
            bs_embed, seq_len, _ = id_embed.shape
            id_embed = id_embed.view(bs_embed, seq_len, -1)
            id_embed = id_embed.to(device='cuda', dtype=torch.bfloat16)

        # 处理控制图像(如果提供)
        if control_image is not None:
            # 转换控制图像为 PIL 格式
            if isinstance(control_image, torch.Tensor):
                if control_image.dim() == 4 and control_image.shape[0] == 1:
                    control_image = control_image.squeeze(0)
                
                control_image_np = (control_image.cpu().numpy() * 255).astype(np.uint8)
                control_image_pil = Image.fromarray(control_image_np)
            else:
                control_image_pil = control_image
            
            control_image_pil = control_image_pil.convert("RGB")
            
            # 在控制图像中检测人脸
            control_face_info = app_640.get(cv2.cvtColor(np.array(control_image_pil), cv2.COLOR_RGB2BGR))
            if len(control_face_info) == 0:
                print("警告: 在控制图像中未检测到人脸。使用黑色图像代替。")
                h, w = control_image_pil.size[::-1]
                out_img = np.zeros([h, w, 3])
                control_image_processed = Image.fromarray(out_img.astype(np.uint8))
            else:
                # 使用最大的人脸
                control_face_info = sorted(control_face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
                control_image_processed = draw_kps(control_image_pil, control_face_info['kps'])
        else:
            # 如果没有提供控制图像,创建黑色图像
            # 使用输入图像的尺寸
            h, w = id_image_pil.size[::-1]
            out_img = np.zeros([h, w, 3])
            control_image_processed = Image.fromarray(out_img.astype(np.uint8))
        
        # 转换处理后的控制图像为张量
        control_image_tensor = torch.from_numpy(np.array(control_image_processed)).float() / 255.0
        if control_image_tensor.dim() == 3:
            control_image_tensor = control_image_tensor.unsqueeze(0)

        # 创建 conditioning
        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()
                d['control'] = infusenet
                d['control_apply_to_uncond'] = False
                d['control_prompt_embeds'] = id_embed
                d['control_conditioning_scale'] = infusenet_conditioning_scale
                d['control_guidance_start'] = infusenet_guidance_start
                d['control_guidance_end'] = infusenet_guidance_end
                
                # 添加必要的控制参数
                d['control_hint'] = control_image_tensor if control_image_tensor is not None else None
                d['control_strength'] = infusenet_conditioning_scale
                d['control_start_stop'] = (infusenet_guidance_start, infusenet_guidance_end)
                
                n = [t[0], d]
                c.append(n)
            out.append(c)
        
        return (model, out[0], out[1])


# Node for creating InfiniteYou conditioning parameters
class InfuConditioningParams:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "infusenet": ("FLUX_CONTROLNET",),
                "id_embeds": ("CONTROLNET_EMBEDS",),
                "infusenet_conditioning_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "infusenet_guidance_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "infusenet_guidance_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "create_conditioning"
    CATEGORY = "InfiniteYou"

    def create_conditioning(self, infusenet, id_embeds, infusenet_conditioning_scale, 
                           infusenet_guidance_start, infusenet_guidance_end):
        # Create a conditioning dictionary that ComfyUI can understand
        conditioning = {
            "controlnet": infusenet,
            "controlnet_prompt_embeds": id_embeds,
            "controlnet_conditioning_scale": infusenet_conditioning_scale,
            "control_guidance_start": infusenet_guidance_start,
            "control_guidance_end": infusenet_guidance_end,
        }
        
        return (conditioning,)




class LoadInfuInsightFace:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "provider": (["CPU", "CUDA", "ROCM"], {"default": "CUDA"}),
            },
        }

    RETURN_TYPES = ("INFU_FACEANALYSIS",)
    FUNCTION = "load_insightface"
    CATEGORY = "InfiniteYou"

    def load_insightface(self, provider):
        insightface_path = os.path.join(folder_paths.models_dir, "insightface")
        # 确保路径存在
        if not os.path.exists(insightface_path):
            os.makedirs(insightface_path, exist_ok=True)
            
        # 设置执行提供者
        providers = [f'{provider}ExecutionProvider']
        if provider != "CPU":
            providers.append('CPUExecutionProvider')
            
        # 加载不同尺寸的 face analyzers
        app_640 = FaceAnalysis(
            name='antelopev2',
            root=insightface_path,
            providers=providers
        )
        app_640.prepare(ctx_id=0, det_size=(640, 640))
        
        app_320 = FaceAnalysis(
            name='antelopev2',
            root=insightface_path,
            providers=providers
        )
        app_320.prepare(ctx_id=0, det_size=(320, 320))
        
        app_160 = FaceAnalysis(
            name='antelopev2',
            root=insightface_path,
            providers=providers
        )
        app_160.prepare(ctx_id=0, det_size=(160, 160))
        
        # 加载 ArcFace 模型
        device = 'cuda' if provider in ['CUDA', 'ROCM'] else 'cpu'
        arcface_model = init_recognition_model('arcface', device=device)
        arcface_model = torch.quantization.quantize_dynamic(
            arcface_model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        # 打包所有分析器
        face_analyzers = {
            "app_640": app_640,
            "app_320": app_320,
            "app_160": app_160,
            "arcface_model": arcface_model
        }

        print(f"InsightFace models loaded with {provider} provider")
        return (face_analyzers,)
    


# Node mapping
NODE_CLASS_MAPPINGS = {
    "LoadInfuModel": LoadInfuModel,
    "LoadInfuInsightFace": LoadInfuInsightFace,
    "ApplyInfu": ApplyInfu,
    "InfuConditioningParams": InfuConditioningParams,
}

# Node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadInfuModel": "Load InfiniteYou Model",
    "LoadInfuInsightFace": "Load InsightFace (InfiniteYou)",
    "ApplyInfu": "Apply InfiniteYou",
    "InfuConditioningParams": "InfiniteYou Conditioning Parameters",
}