from torch.utils.data import DataLoader
from src.group_99.data import load_data, CustomDataset

data,_,_,_ = load_data()
dataset = CustomDataset(data)
for workers in [0, 1, 2, 4, 8]:
    print(f"Testing with num_workers={workers}")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=workers)
    try:
        for batch in dataloader:
            pass
        print(f"Success with num_workers={workers}")
    except Exception as e:
        print(f"Failed with num_workers={workers}: {e}")
