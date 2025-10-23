from torch.utils.cpp_extension import load
import torch
ext=load(
    name="msgpass_ext",
    sources=["message_passing.cpp", "message_passing.cu"],
    extra_include_paths=torch.utils.cpp_extension.include_paths(),  # helps if needed
    verbose=True
)
message_passing = ext.message_passing