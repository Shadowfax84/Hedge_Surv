import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
import numpy as np  # Import NumPy

# Define transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class SimpleNet(nn.Module):
    def __init__(self, num_classes=1):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(32 * 32 * 3, num_classes)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = self.fc(x)
        return x


def train_with_dp(img_array, num_classes=1, num_epochs=5):
    """Train the model with differential privacy."""

    # Convert img_array (raw byte data) to a NumPy array of uint8
    # Assuming img_array is raw byte data representing an image
    # Adjust dtype and reshape as needed for your specific data

    try:
        numerical_img_array = np.frombuffer(img_array, dtype=np.uint8)
    except ValueError as e:
        print(f"Error converting byte data: {e}")
        return None  # Or raise the exception, depending on how you want to handle errors

    try:
        # -1 infers the number of images
        numerical_img_array = numerical_img_array.reshape(-1, 32, 32, 3)
    except ValueError as e:
        print(f"Error reshaping image data: {e}. Check image dimensions.")
        return None

    # Convert img_array to a PyTorch tensor and create a DataLoader
    # Ensure it's a float tensor
    tensor_data = torch.tensor(numerical_img_array).float()
    tensor_labels = torch.zeros(tensor_data.size(
        0), dtype=torch.long)  # Dummy labels

    train_loader = DataLoader(
        list(zip(tensor_data, tensor_labels)), batch_size=32, shuffle=True)

    model = SimpleNet(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    # Initialize Privacy Engine
    privacy_engine = PrivacyEngine()

    # Attach the privacy engine
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=1.3,
        max_grad_norm=1.0,
    )

    criterion = nn.CrossEntropyLoss()

    print("Starting training with differential privacy...")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Print progress every few batches
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Loss={loss.item():.4f}")

        # Calculate and print the privacy budget spent
        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        print(f"Epoch {epoch + 1}: ε = {epsilon:.2f}, δ = 1e-5")

    print("Training completed.")

    return model.state_dict()  # Return updated model parameters
