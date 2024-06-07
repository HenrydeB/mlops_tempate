import torch
import torchvision.models as models
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

model = models.resnet18()  # neural network model
inputs = torch.randn(5, 3, 224, 224)  # simple tensor model
with profile(
    activities=[ProfilerActivity.CPU], record_shapes=True, on_trace_ready=tensorboard_trace_handler("./tensorboard")
) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

# visualization
# prof.export_chrome_trace("pytorch_trace.json") single line

# with profile(...) as prof:
#    for i in range (10):
#       model(inputs)
#      prof.step()
