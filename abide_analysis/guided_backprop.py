
import importlib
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import pickle 
import matplotlib.pyplot as plt
import numpy as np

# Assuming sys.argv[1] is the module name
module_name = sys.argv[1]  # Example: "my_model"
class_name = "AgePredictionCNN"  # The class you want to import

try:
    # Dynamically import the module
    module = importlib.import_module(module_name)
    
    # Dynamically get the class
    AgePredictionCNN = getattr(module, class_name)
    
    print(f"Successfully imported {class_name} from {module_name}.")

except ImportError:
    print(f"Module {module_name} could not be imported.")
except AttributeError:
    print(f"{class_name} does not exist in {module_name}.")


import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# 3. Load trained model
# ----------------------

# Example shape from training
input_shape = (1, 160, 768)  # (channels, H, W)

model = AgePredictionCNN(input_shape=input_shape).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)


load_saved = sys.argv[2] # "last, "best"
if load_saved != "none":
    with open(f"model_dumps/mix/{sys.argv[1]}_best_model_with_metadata.pkl", "rb") as f:
        checkpoint = pickle.load(f)
    best_loss = checkpoint["loss"]

    # Load the checkpoint
    with open(f"model_dumps/mix/{sys.argv[1]}_{load_saved}_model_with_metadata.pkl", "rb") as f:
        checkpoint = pickle.load(f)

    # Restore model and optimizer state
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    # Restore RNG states
    torch.set_rng_state(checkpoint["t_rng_st"])
    np.random.set_state(checkpoint["n_rng_st"])
    if torch.cuda.is_available() and checkpoint["cuda_rng_st"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_st"])

    # Retrieve metadata
    start_epoch = checkpoint["epoch"] + 1
    loaded_loss = checkpoint["loss"]

    print(f"Loaded model from epoch {start_epoch} with validation loss {loaded_loss:.4f}, best loss {best_loss:.4f}")


model.eval()


# ----------------------
# 2. Guided Backprop helper
# ----------------------
class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.forward_relu_outputs = []
        self.hooks = []

        def relu_forward_hook(module, input, output):
            self.forward_relu_outputs.append(output)

        def relu_backward_hook(module, grad_input, grad_output):
            # Only allow positive gradients to flow back
            forward_output = self.forward_relu_outputs.pop()
            forward_mask = (forward_output > 0).float()
            modified_grad_output = grad_output[0].clamp(min=0.0)
            return (modified_grad_output * forward_mask,)

        # Replace all ReLUs/SiLUs with hooks
        for module in self.model.modules():
            if isinstance(module, nn.ReLU) or isinstance(module, nn.SiLU):
                self.hooks.append(module.register_forward_hook(relu_forward_hook))
                self.hooks.append(module.register_full_backward_hook(relu_backward_hook))

    def generate(self, input_tensor, sex_tensor):
        input_tensor.requires_grad = True

        output = self.model(input_tensor, sex_tensor)  # forward
        pred = output.squeeze()

        self.model.zero_grad()
        pred.backward(retain_graph=True)

        # Guided gradients wrt input
        guided_grads = input_tensor.grad.detach().cpu().numpy()[0, 0]  # (160,768)

        # Normalize for visualization
        guided_grads = (guided_grads - guided_grads.min()) / (guided_grads.max() - guided_grads.min() + 1e-8)
        return guided_grads, pred.item()

    def close(self):
        for hook in self.hooks:
            hook.remove()


# ----------------------
# 3. Guided Backprop instantiation
# ----------------------
guided_bp = GuidedBackprop(model)

# ----------------------
# 4. Example input (same as GradCAM)
# ----------------------
features = torch.randn(1, 1, 160, 768).to(device)  # fake example
sex = torch.tensor([1.0]).to(device)               # male/female encoding

# ----------------------
# 5. Generate Guided Backprop
# ----------------------
guided_map, pred = guided_bp.generate(features, sex)
print(f"Predicted Age: {pred:.2f}")

# ----------------------
# 6. Plot guided backprop saliency (rotated for better orientation)
# ----------------------
plt.figure(figsize=(6, 12))

# Transpose to swap axes: slices on x-axis, embedding on y-axis
plt.imshow(guided_map.T, cmap="jet", aspect="auto", origin="lower")

plt.colorbar(label="Guided Backprop Gradient")
plt.title("Guided Backpropagation Saliency Map (rotated)")
plt.xlabel("Sagittal slice index (160)")
plt.ylabel("Embedding dimension (768)")
# Save the figure
plt.savefig(f"gbp_{sys.argv[1]}_{sys.argv[2]}.png", bbox_inches='tight', dpi=300)
plt.show()

