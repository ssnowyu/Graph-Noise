import torch


def sample_mask(mask: torch.Tensor, percentage: float):
    # print(mask)
    one_indices = torch.nonzero(mask == 1).squeeze()
    # print(one_indices)
    num_samples = int(percentage * len(one_indices))
    selected_indices = torch.randperm(len(one_indices))[:num_samples]
    new_mask = torch.zeros_like(mask)
    new_mask[one_indices[selected_indices]] = 1
    return new_mask


if __name__ == "__main__":
    mask = torch.tensor([1, 0, 0, 1, 1, 1, 0])
    new_mask = sample_mask(mask, 0.5)
    print(new_mask)
