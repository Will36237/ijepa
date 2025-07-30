import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from logging import getLogger
import os
logger = getLogger()

def make_time_series(
    transform=None,
    batch_size=32,
    collator=None,
    pin_mem=True,
    num_workers=0,
    world_size=1,
    rank=0,
    root_path=None,
    data_file=None,
    training=True,
    drop_last=True,
    window_size=20,  # 5 tiếng (20 điểm cho M15)
    segment_size=5   # Đoạn nhỏ (5 điểm = 75 phút)
):

    dataset = TimeSeriesDataset(
        root_path=root_path,
        data_file=data_file,
        transform=transform,
        train=training,
        window_size=window_size,
        segment_size=segment_size
    )
    logger.info('TimeSeries dataset created')
    
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    data_loader = DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )
    logger.info('TimeSeries unsupervised data loader created')

    return dataset, data_loader, dist_sampler

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        root_path,
        data_file,
        transform=None,
        train=True,
        window_size=20,
        segment_size=5
    ):
       
        self.transform = transform
        self.window_size = window_size
        self.segment_size = segment_size
        self.num_segments = window_size // segment_size
        self.train = train
        
        data_path = os.path.join(root_path, data_file)
        logger.info(f'Data path: {data_path}')
        
        self.data = pd.read_csv(data_path)
        self.features = self.data.select_dtypes(include=[np.number]).values
        
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)
        
        self.windows = self.create_windows(self.scaled_features)
        logger.info('Initialized TimeSeriesDataset')
    
    def create_windows(self, data):
        """
        Chia dữ liệu thành các đoạn lớn.

        Args:
            data (np.ndarray): Dữ liệu chuẩn hóa, shape [num_samples, num_features].

        Returns:
            np.ndarray: Mảng các đoạn, shape [num_windows, window_size, num_features].
        """
        windows = []
        stride = 1 if self.train else self.window_size
        for i in range(0, len(data) - self.window_size + 1, stride):
            windows.append(data[i:i + self.window_size])
        return np.array(windows)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        if self.transform:
            window = self.transform(window)
        
        window = torch.tensor(window, dtype=torch.float32)
        return window  
