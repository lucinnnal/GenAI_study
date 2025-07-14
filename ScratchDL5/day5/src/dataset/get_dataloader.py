from torch.utils.data import DataLoader
from src.dataset.get_dataset import get_train_dataset

def get_dataloader(args):
    train_dataset = get_train_dataset()
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    return dataloader