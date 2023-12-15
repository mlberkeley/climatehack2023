import torch
class NonHrvDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path="/data/climatehack/official_dataset/nonhrv"):
        self.data_path = data_path
        self.index_map = {}
        self.month_map = {0: 0}

        def xr_open(path):
            return xr.open(
                    path,
                    engine="zarr",
                    consolidated=True,
                    )

        def verify(real_idx):
            return True

        year = 2021
        self.data_2021 = [xr_open(f"/data/climatehack/official_dataset/nonhrv/{year}/{month}.zarr.zip") for month in range(1,13)]

    def __len__(self):
        return sum([x.shape[0] for x in self.data_2021])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
