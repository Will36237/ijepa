import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from datasets import load_dataset, DatasetDict  # Import load_dataset
import pandas as pd
from logging import getLogger
logger = getLogger()

def split_data(hf_dataset, split_ratios, features_list):
    df = hf_dataset.to_pandas()
    grouped = df.groupby(['symbol', 'timeframe'])
    train_data, test_data, val_data = [], [], []
    for _, group in grouped:
        n = len(group)
        train_end = int(n * split_ratios[0])
        test_end = train_end + int(n * split_ratios[1])
        train_data.append(group.iloc[:train_end])
        test_data.append(group.iloc[train_end:test_end])
        val_data.append(group.iloc[test_end:])
    train_df = pd.concat(train_data)
    test_df = pd.concat(test_data)
    val_df = pd.concat(val_data)
    return DatasetDict({'train': Dataset.from_pandas(train_df), 'test': Dataset.from_pandas(test_df), 'validation': Dataset.from_pandas(val_df)})

class TimeSeriesDataset(Dataset):
    def __init__(self, hf_split, window_size, segment_size, num_features, future_steps, features_list):
        self.data = hf_split
        self.window_size = window_size
        self.segment_size = segment_size
        self.num_features = num_features
        self.future_steps = future_steps
        self.features_list = features_list
        self.groups = [f"{s}_{t}" for s, t in zip(self.data['symbol'], self.data['timeframe'])]

    def __len__(self):
        return len(self.data) - self.window_size - self.future_steps

    def __getitem__(self, idx):
        window_data = {f: self.data[idx:idx+self.window_size][f] for f in self.features_list}
        window = torch.tensor(pd.DataFrame(window_data).values, dtype=torch.float)  # [window_size, num_features]
        labels = torch.tensor(self.data[idx+self.window_size:idx+self.window_size+self.future_steps]['zigzag_small'], dtype=torch.long)
        group_id = self.groups[idx]
        return window, labels, group_id

def make_time_series(dataset_name, window_size, segment_size, batch_size, training, collator, pin_mem, num_workers, world_size, rank, **kwargs):
    hf_dataset = load_dataset(dataset_name)  # Load full dataset
    if isinstance(hf_dataset, DatasetDict):
        full_df = pd.concat([split.to_pandas() for split in hf_dataset.values()])
        hf_dataset = Dataset.from_pandas(full_df)
    else:
        pass

    split_ratios = kwargs['split_ratios']  
    features_list = kwargs['features_list']
    future_steps = kwargs['future_steps']
    num_features = kwargs['num_features']

    split_dataset = split_data(hf_dataset, split_ratios, features_list)
    split = 'train' if training else 'test'
    ts_dataset = TimeSeriesDataset(split_dataset[split], window_size, segment_size, num_features, future_steps, features_list)
    sampler = DistributedSampler(ts_dataset, num_replicas=world_size, rank=rank, shuffle=training)
    data_loader = DataLoader(ts_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_mem, collate_fn=collator)
    return ts_dataset, data_loader, sampler
