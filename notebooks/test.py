from dgl.data import CoraGraphDataset


dataset = CoraGraphDataset(reverse_edge=True)
g = dataset[0]
print(g.num_edges())
