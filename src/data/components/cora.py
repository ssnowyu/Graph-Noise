import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from dgl.data import DGLDataset, CoraGraphDataset
import torch
from src.utils.data_utils import sample_mask


class CoraDataset(DGLDataset):
    def __init__(
        self,
        raw_dir=None,
        save_dir=None,
        reverse_edge: bool = True,
        noise_rate: float = 0.0,
        sigma: float = 1e-2,
    ):
        self.noise_rate = noise_rate
        self.reverse_edge = reverse_edge
        self.sigma = sigma
        super().__init__(name="Cora", raw_dir=raw_dir, save_dir=save_dir)

    def download(self):
        CoraGraphDataset(raw_dir=self.raw_dir)

    def process(self):
        g = CoraGraphDataset(raw_dir=self.raw_dir, reverse_edge=self.reverse_edge)[0]
        feat = g.ndata["feat"]
        # add noises to feature
        noise = torch.randn_like(feat) * self.sigma
        train_mask = g.ndata["train_mask"]
        val_mask = g.ndata["val_mask"]
        test_mask = g.ndata["test_mask"]

        train_noise_mask = sample_mask(train_mask, self.noise_rate).int()
        val_noise_mask = sample_mask(val_mask, self.noise_rate).int()
        test_noise_mask = sample_mask(test_mask, self.noise_rate).int()

        noise_mask = train_noise_mask + val_noise_mask + test_noise_mask

        num_words = (feat > 0).sum(dim=1)
        feat = num_words.unsqueeze(1).repeat(1, feat.size(-1)) * feat

        g.ndata["feat"] = feat + noise * noise_mask.unsqueeze(1).repeat(
            1, feat.size(-1)
        )

        self.graphs = [g]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.graphs[idx]


if __name__ == "__main__":
    dataset = CoraDataset(raw_dir="data/cora/raw", noise_rate=0.6, sigma=0.2)
    g = dataset[0]
    feat = g.ndata["feat"]
    # print(feat.sum(1))
