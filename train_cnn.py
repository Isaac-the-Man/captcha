import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from einops import rearrange
import os
from src.utils.dataset import CaptchaDataset
from src.model.cnn import CaptchaModelCNN
from src.utils.utils import batchOnehotEncodeLabel, batchOnehotDecodeLabel


from torchvision.transforms import v2

transforms = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.ToDtype(torch.float32, scale=True),
])

train_ds = CaptchaDataset('dataset', split='train', transform=transforms)
valid_ds = CaptchaDataset('dataset', split='valid', transform=transforms)
test_ds = CaptchaDataset('dataset', split='test', transform=transforms)

EPOCHS = 12000
BATCH_SIZE = 1024
LEARNING_RATE = 0.0001
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# attempt to load from checkpoint
last_chkpt = None
for path in os.listdir('checkpoints/train_cnn'):
    if path.endswith('.pth'):
        epoch = int(path.split('_')[-1].split('.')[0])
        if last_chkpt is None or epoch > last_chkpt:
            last_chkpt = epoch
model = None
if last_chkpt is not None:
    checkpoint = torch.load(f'checkpoints/train_cnn/{path}')
    model = CaptchaModelCNN().to(device)
    model.load_state_dict(checkpoint)
    print(f'Loaded from checkpoint: {last_chkpt}')
else:
    model = CaptchaModelCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

from tqdm import tqdm

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

if last_chkpt is None:
    last_chkpt = 0
for epoch in range(last_chkpt, EPOCHS):
    print(f'Epoch: {epoch}')
    # training loop
    model.train()
    total_loss = 0
    for i, (x, y) in tqdm(enumerate(train_dl), total=len(train_dl)):
        x, y = x.to(device), batchOnehotEncodeLabel(y).to(device)
        optimizer.zero_grad()

        # forward pass
        output = model(x)

        output = rearrange(output.reshape(-1,5,36), 'b n c -> (b n) c')

        # compute loss
        loss = criterion(output, y.view(-1).long())
        total_loss += loss.item()

        # back prop
        loss.backward()
        optimizer.step()
    
    print(f'Epoch: {epoch}, Average Loss: {total_loss / len(train_dl)}')

    # validation loop
    if epoch % 20 == 0:
        with torch.no_grad():
            model.eval()
            total = 0
            correct = 0
            valid_total_loss = 0
            for i, (x, yStr) in enumerate(valid_dl):
                x, y = x.to(device), batchOnehotEncodeLabel(yStr).to(device)

                # forward pass
                output = model(x)

                output = rearrange(output.reshape(-1,5,36), 'b n c -> (b n) c')

                # compute loss
                loss = criterion(output, y.view(-1).long())
                valid_total_loss += loss.item()

                # get predicted labels
                predStr = batchOnehotDecodeLabel(output.reshape(-1, 5, 36))

                # compute accuracy
                total += len(yStr)
                for pred, label in zip(predStr, yStr):
                    if pred == label:
                        correct += 1
            print(f'Epoch: {epoch}, Validation Accuracy: {correct / total}')
            print(f'Epoch: {epoch}, Validation Loss: {valid_total_loss / len(valid_dl)}')
            # save to checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            os.makedirs('checkpoints/train_cnn', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/train_cnn/checkpoint_{epoch}.pth')
