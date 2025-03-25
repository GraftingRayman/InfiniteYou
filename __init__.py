
from .nodes.infu_nodes import LoadInfuModel, ApplyInfu, InfuConditioningParams, LoadInfuInsightFace



# Register only GRSeed as a node
NODE_CLASS_MAPPINGS = {
    "LoadInfuModel": LoadInfuModel,
    "LoadInfuInsightFace": LoadInfuInsightFace,
    "ApplyInfu": ApplyInfu,
    "InfuConditioningParams": InfuConditioningParams,


}



__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


