import torch
import numpy as np
from typing import List, Dict
from crp.concepts import Concept

class TimeSeriesConcept(Concept):
    """
    Concept Class for 1D CNN layers processing time series data
    Adapted from ChannelConcept for temporal sequences
    """

    @staticmethod
    def mask(batch_id: int, concept_ids: List, layer_name=None):
        """
        Wrapper for 1D CNN gradient masking
        
        Parameters:
        ----------
        batch_id: int - Batch dimension index
        concept_ids: list - Channel indices for time series (0, 1, 2 for X,Y,Z)
        """
        def mask_fct(grad):
            mask = torch.zeros_like(grad[batch_id])
            mask[concept_ids] = 1
            grad[batch_id] = grad[batch_id] * mask
            return grad
        return mask_fct

    @staticmethod
    def mask_rf(batch_id: int, c_n_map: Dict[int, List], layer_name=None):
        """
        Temporal receptive field masking for specific time segments
        
        Parameters:
        ----------
        c_n_map: dict - {channel_id: [temporal_indices]}
        """
        def mask_fct(grad):
            grad_shape = grad.shape  # [batch, channels, time]
            mask = torch.zeros_like(grad[batch_id])
            
            for channel in c_n_map:
                temporal_indices = c_n_map[channel]
                mask[channel, temporal_indices] = 1
                
            grad[batch_id] = grad[batch_id] * mask
            return grad
        return mask_fct

    def get_rf_indices(self, output_shape, layer_name=None):
        """Get temporal indices for receptive field analysis"""
        if len(output_shape) == 1:
            return [0]
        else:
            # For 1D CNN: output_shape = [channels, time_length]
            return np.arange(0, output_shape[-1])  # Time dimension

    def attribute(self, relevance, mask=None, layer_name: str = None, abs_norm=True):
        """
        Compute channel-wise attribution for time series
        
        Parameters:
        ----------
        relevance: torch.Tensor - Shape [batch, channels, time]
        """
        if isinstance(mask, torch.Tensor):
            relevance = relevance * mask

        # Sum over temporal dimension to get channel importance
        rel_l = torch.sum(relevance, dim=-1)  # [batch, channels]

        if abs_norm:
            rel_l = rel_l / (torch.abs(rel_l).sum(-1).view(-1, 1) + 1e-10)

        return rel_l

    def reference_sampling(self, relevance, layer_name: str = None, max_target: str = "sum", abs_norm=True):
        """
        Find most relevant temporal segments and channels
        
        Parameters:
        ----------
        max_target: 'sum' or 'max' - How to aggregate temporal relevance
        """
        # Shape: [batch, channels, time]
        rel_temporal = relevance.view(*relevance.shape)
        
        # Find most relevant time point per channel
        rf_temporal = torch.argmax(rel_temporal, dim=-1)  # [batch, channels]

        # Channel importance calculation
        if max_target == "sum":
            rel_channels = torch.sum(rel_temporal, dim=-1)  # [batch, channels]
        elif max_target == "max":
            rel_channels = torch.max(rel_temporal, dim=-1)[0]  # [batch, channels]
        else:
            raise ValueError("'max_target' supports only 'max' or 'sum'.")

        if abs_norm:
            rel_channels = rel_channels / (torch.abs(rel_channels).sum(-1).view(-1, 1) + 1e-10)
        
        # Sort channels by relevance
        d_ch_sorted = torch.argsort(rel_channels, dim=0, descending=True)
        rel_ch_sorted = torch.gather(rel_channels, 0, d_ch_sorted)
        rf_ch_sorted = torch.gather(rf_temporal, 0, d_ch_sorted)

        return d_ch_sorted, rel_ch_sorted, rf_ch_sorted