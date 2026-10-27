"""Small temporal model. Returns logits; use BCEWithLogitsLoss during training."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class FacialGRU(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, sequences, lengths):
        """sequences: [batch,time,features]; lengths: positive integer [batch].

        Inputs must be finite, preprocessed with training-only statistics. Padding
        after each valid length is ignored. Batches need not be sorted by length.
        """
        if sequences.ndim != 3 or sequences.shape[0] == 0 or sequences.shape[2] != self.gru.input_size:
            raise ValueError("Expected nonempty [batch,time,input_size] sequences.")
        lengths = torch.as_tensor(lengths)
        if lengths.ndim != 1 or lengths.numel() != sequences.shape[0]:
            raise ValueError("Provide one valid length per sequence.")
        if lengths.is_floating_point() or lengths.is_complex() or lengths.dtype == torch.bool:
            raise ValueError("Lengths must be integers.")
        if (lengths < 1).any() or (lengths > sequences.shape[1]).any():
            raise ValueError("Lengths must be between 1 and the padded time dimension.")
        if not torch.isfinite(sequences).all():
            raise ValueError("Preprocess missing or infinite sequence values first.")
        packed = pack_padded_sequence(sequences, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.output(self.dropout(hidden[-1])).squeeze(-1)
