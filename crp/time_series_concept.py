import torch
import numpy as np
from typing import List, Dict
from crp.concepts import Concept, ChannelConcept

class TimeSeriesConcept(ChannelConcept):
    """
    Concept Class specifically for 1D time series data with multiple channels
    """
    
    @staticmethod
    def mask(batch_id: int, concept_ids: List, layer_name=None):
        """
        Creates a mask function that preserves channel-specific attributions for time series data
        
        Parameters:
        ----------
        batch_id: int
            Specifies the batch dimension in the torch.Tensor.
        concept_ids: list of integer values
            Integer lists corresponding to channel or concept indices.
            
        Returns:
        --------
        callable function that modifies the gradient while preserving channel structure
        """
        def mask_fct(grad):
            # Create initial mask filled with zeros
            mask = torch.zeros_like(grad[batch_id])
            
            # Apply mask to specified concepts/channels
            mask[concept_ids] = 1
            
            # Apply the mask to the gradient
            grad[batch_id] = grad[batch_id] * mask
            
            return grad
        return mask_fct
    
    def attribute(self, relevance, mask=None, layer_name: str = None, abs_norm=True):
        """
        Attribute relevance to concepts, preserving the channel structure for time series data
        
        Parameters:
        ----------
        relevance: torch.Tensor
            The relevance tensor from backpropagation
        mask: torch.Tensor or None
            Optional mask to apply to the relevance
        layer_name: str
            Name of the layer (unused but kept for API compatibility)
        abs_norm: bool
            Whether to normalize by absolute sum
            
        Returns:
        --------
        torch.Tensor: Channel-specific attribution scores
        """
        if isinstance(mask, torch.Tensor):
            relevance = relevance * mask

        # For time series: keep channel dimension (axis 1) separate from time dimension
        # Sum over the time dimension (axis 2) to get per-channel relevance
        rel_l = torch.sum(relevance, dim=2)

        if abs_norm:
            # Normalize while preserving channel information
            abs_sum = torch.sum(torch.abs(rel_l), dim=1, keepdim=True) + 1e-10
            rel_l = rel_l / abs_sum

        return rel_l

class ChannelTimeSeriesConcept(TimeSeriesConcept):
    """
    An extension of TimeSeriesConcept that allows specifying different concepts per channel
    """
    
    @staticmethod
    def mask(batch_id: int, concept_channel_map: Dict[int, List], layer_name=None):
        """
        Creates a mask function that preserves channel-specific attributions for time series data
        
        Parameters:
        ----------
        batch_id: int
            Specifies the batch dimension in the torch.Tensor.
        concept_channel_map: dict with int keys and list values
            Keys are channel indices and values are time points to focus on
            
        Returns:
        --------
        callable function that modifies the gradient while preserving channel structure
        """
        def mask_fct(grad):
            # Create initial mask filled with zeros
            mask = torch.zeros_like(grad[batch_id])
            
            # Apply mask to specified channels and time points
            for channel, time_points in concept_channel_map.items():
                if isinstance(time_points, list) and len(time_points) > 0:
                    # Time points specified for this channel
                    mask[channel, time_points] = 1
                else:
                    # Use entire channel
                    mask[channel, :] = 1
            
            # Apply the mask to the gradient
            grad[batch_id] = grad[batch_id] * mask
            
            return grad
        return mask_fct
