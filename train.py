from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from T1mra_dataset import T1w2MraDataset
from UNet import UNet
import torch.optim as optim

# train_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])

train_transform = transforms.ToTensor()


dataset = T1w2MraDataset("/Users/asagilmore/src/t1-mra/processed-data/T1W",
                         "/Users/asagilmore/src/t1-mra/processed-data/MRA",
                         transform=train_transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
model = UNet(1, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 25  # Number of epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        optimizer.zero_grad()
        outputs = model(images.float())
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}")