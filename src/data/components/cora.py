import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from dgl.data import DGLDataset, CoraGraphDataset
import torch

# from src.utils.data_utils import sample_mask
from dgl.sampling import global_uniform_negative_sampling
import dgl


class CoraDataset(DGLDataset):
    def __init__(
        self,
        raw_dir=None,
        save_dir=None,
        reverse_edge: bool = True,
        force_reload: bool = True,
        noise_type: str = "none",  # ["none", "feat", "missing-edge", "redundant-edge", "error-edge"]
        feat_noise_rate: float = 0.0,
        feat_sigma: float = 0.0,
        edge_rate: float = 0.0,
    ):
        self.reverse_edge = reverse_edge

        self.noise_type = noise_type
        self.feat_noise_rate = feat_noise_rate
        self.feat_sigma = feat_sigma
        self.edge_rate = edge_rate
        super().__init__(
            name="Cora", raw_dir=raw_dir, save_dir=save_dir, force_reload=force_reload
        )

    def download(self):
        CoraGraphDataset(raw_dir=self.raw_dir, reverse_edge=False)

    def process(self):
        g = CoraGraphDataset(
            raw_dir=self.raw_dir, reverse_edge=False, force_reload=True
        )[0]

        print(f"num_edges: {g.num_edges()}")

        # Add noise
        if self.noise_type == "none":
            pass
        elif self.noise_type == "feat":
            print("add feat noise")
            g = self.add_feat_noise(g)
        elif self.noise_type == "missing-edge":
            print("add missing edge noise")
            g = self.add_missing_edge_noise(g)
        elif self.noise_type == "redundant-edge":
            print("add redundant edge noise")
            g = self.add_redundant_edge_noise(g)
        elif self.noise_type == "error-edge":
            print("add error edge noise")
            g = self.add_error_edge_noise(g)

        self.graphs = [g]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.graphs[idx]

    def add_feat_noise(self, g):
        num_noisy_rows = int(g.ndata["feat"].size(0) * self.feat_noise_rate)
        noisy_rows = torch.randperm(g.ndata["feat"].size(0))[:num_noisy_rows]
        noise = torch.randn((num_noisy_rows, g.ndata["feat"].size(1))) * self.feat_sigma
        noisy_data = torch.zeros(g.ndata["feat"].size())
        noisy_data[noisy_rows] = noise

        g.ndata["feat"] = g.ndata["feat"] + noisy_data
        g = dgl.to_bidirected(g)

        return g

    def add_missing_edge_noise(self, g):
        num_edges = g.num_edges()
        remove_edge_index = torch.randperm(num_edges)[: int(num_edges * self.edge_rate)]
        g = dgl.remove_edges(g, remove_edge_index)
        g = dgl.to_bidirected(g)
        return g

    def add_redundant_edge_noise(self, g):
        num_edges = g.num_edges()
        negative_srcs, negative_dsts = global_uniform_negative_sampling(
            g, int(num_edges * self.edge_rate)
        )

        g = dgl.add_edges(g, negative_srcs, negative_dsts)

        g = dgl.to_bidirected(g)
        return g

    def add_error_edge_noise(self, g):
        num_edges = g.num_edges()
        remove_edge_index = torch.randperm(num_edges)[: int(num_edges * self.edge_rate)]
        negative_srcs, negative_dsts = global_uniform_negative_sampling(
            g, int(num_edges * self.edge_rate)
        )
        g = dgl.remove_edges(g, remove_edge_index)
        g = dgl.add_edges(g, negative_srcs, negative_dsts)
        g = dgl.to_bidirected(g)
        return g


if __name__ == "__main__":
    dataset = CoraDataset(raw_dir="data/cora/raw", feat_noise_rate=0.6, feat_sigma=0.2)
    g = dataset[0]
    feat = g.ndata["feat"]
    # print(feat.sum(1))
