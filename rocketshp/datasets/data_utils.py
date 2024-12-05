import torch
import numpy as np

def _stack_variable_length_tensors(
    sequences: list[torch.Tensor],
    constant_value: int | float = 0,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Automatically stack tensors together, padding variable lengths with the
    value in constant_value. Handles an arbitrary number of dimensions.

    From https://github.com/evolutionaryscale/esm/blob/main/esm/utils/misc.py#L126

    Examples:
        >>> tensor1, tensor2 = torch.ones([2]), torch.ones([5])
        >>> stack_variable_length_tensors(tensor1, tensor2)
        tensor of shape [2, 5]. First row is [1, 1, 0, 0, 0]. Second row is all ones.

        >>> tensor1, tensor2 = torch.ones([2, 4]), torch.ones([5, 3])
        >>> stack_variable_length_tensors(tensor1, tensor2)
        tensor of shape [2, 5, 4]
    """
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype
    device = sequences[0].device

    array = torch.full(shape, constant_value, dtype=dtype, device=device)
    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array

def _unstack_variable_length_tensors(stacked, lengths) -> list[torch.Tensor]:
    """
    Unstack a variable length tensor given the lengths of the sequences
    """
    return [stacked[i, :lengths[i]] for i in range(len(lengths))]

def _unstack_variable_size_squareforms(stacked, lengths):
        
    return [stacked[i, :sl, :sl] for i, sl in enumerate(lengths)]

def _get_seq_lengths(sequences: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

def _dict_collate_fn(batch):
    """
    Custom collate function for batching variable length sequences.
    
    Args:
        batch: List of tuples (features_dict, labels_dict) from the dataset
        
    Returns:
        features_dict: Dictionary of padded feature tensors
        labels_dict: Dictionary of label tensors
        lengths: Tensor of sequence lengths
    """
    # Separate features and labels
    features, labels = zip(*batch)
    
    # Get sequence lengths (assuming all features have same length)
    lengths = torch.tensor([next(iter(feat.values())).shape[0] for feat in features])
    
    # Initialize output dictionaries
    padded_features = {}
    padded_labels = {}
    
    # Pad and stack features
    for key in features[0].keys():
        sequences = [feat[key] for feat in features]
        padded_features[key] = _stack_variable_length_tensors(sequences)
    
    # Stack labels (assuming they don't need padding)
    for key in labels[0].keys():
        label_tensors = [label[key] for label in labels]
        padded_labels[key] = _stack_variable_length_tensors(label_tensors)
    
    return padded_features, padded_labels, lengths
