from torch import nn
from cs336_basics.model import BasicsTransformerLM

def analyze_model_parameters(model: nn.Module) -> dict:
    total_params = 0
    trainable_params = 0
    total_bytes = 0

    for param in model.parameters():
        num_params = param.numel()

        total_params += num_params
        total_bytes += num_params * param.element_size()

        if param.requires_grad:
            trainable_params += num_params

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "parameter_memory_mib": total_bytes / (1024 ** 2),
        "parameter_memory_gib": total_bytes / (1024 ** 3),
    }


if __name__ == "__main__":
    model = BasicsTransformerLM(
        vocab_size=1000,
        context_length=64,
        d_model=768,
        d_ff=3072,
        num_layers=12,
        num_heads=12,
    )
    print(analyze_model_parameters(model))
