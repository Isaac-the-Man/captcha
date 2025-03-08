import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import os
from src.utils.dataset import CaptchaDataset
from src.model.vit import CaptchaModelViT
from src.utils.utils import batchOnehotEncodeLabel, batchDecodeCTCOutput
from tqdm import tqdm


from torchvision.transforms import v2

transforms = v2.Compose([
    v2.Grayscale(num_output_channels=1),
    v2.ToDtype(torch.float32, scale=True),
])

train_ds = CaptchaDataset('dataset', split='train', transform=transforms)
valid_ds = CaptchaDataset('dataset', split='valid', transform=transforms)
test_ds = CaptchaDataset('dataset', split='test', transform=transforms)

EPOCHS = 3000
BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
LEARNING_RATE_END = 1e-6
CHKPT_DIR = 'checkpoints/train_vit'
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# attempt to load from checkpoint
os.makedirs(CHKPT_DIR, exist_ok=True)
last_chkpt = None
for path in os.listdir(CHKPT_DIR):
    if path.endswith('.pth'):
        epoch = int(path.split('_')[-1].split('.')[0])
        if last_chkpt is None or epoch > last_chkpt:
            last_chkpt = epoch
model = CaptchaModelViT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, eta_min=LEARNING_RATE_END)
if last_chkpt is not None:
    checkpoint = torch.load(os.path.join(CHKPT_DIR, f'checkpoint_{last_chkpt}.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f'Loaded from checkpoint: {last_chkpt}')

criterion = nn.CTCLoss()
output_seq_len = torch.ones(BATCH_SIZE, dtype=int, device=device) * 18
target_seq_len = torch.ones(BATCH_SIZE, dtype=int, device=device) * 5

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
        x, y = x.to(device), batchOnehotEncodeLabel(y).to(device) + 1

        optimizer.zero_grad()

        # forward pass
        output = model(x) # (batch, 18, 37)

        # compute loss
        output = F.log_softmax(output.permute(1, 0, 2), dim=2) # (18, batch, 37)
        loss = criterion(output, y, output_seq_len[:x.shape[0]], target_seq_len[:x.shape[0]])
        total_loss += loss.item()

        # back prop
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    print(f'Epoch: {epoch}, Average Loss: {total_loss / len(train_dl)}, LR: {scheduler.get_last_lr()[0]}')

    # validation loop
    if epoch % 20 == 0:
        with torch.no_grad():
            model.eval()
            total = 0
            correct = 0
            valid_total_loss = 0
            for i, (x, yStr) in enumerate(valid_dl):
                x, y = x.to(device), batchOnehotEncodeLabel(yStr).to(device) + 1

                # forward pass
                output = model(x) # (batch, 18, 37)
                # get predicted labels
                predStr = batchDecodeCTCOutput(output)

                # compute loss
                output = F.log_softmax(output.permute(1, 0, 2), dim=2) # (18, batch, 37)
                loss = criterion(output, y, output_seq_len[:x.shape[0]], target_seq_len[:x.shape[0]])
                valid_total_loss += loss.item()

                # compute accuracy
                total += len(yStr)
                for pred, label in zip(predStr, yStr):
                    if pred == label:
                        correct += 1
            print(f'Epoch: {epoch}, Validation Accuracy: {correct / total}')
            print(f'Epoch: {epoch}, Validation Loss: {valid_total_loss / len(valid_dl)}')
            # save to checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(CHKPT_DIR, f'checkpoint_{epoch}.pth'))
