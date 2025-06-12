import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
import seaborn as sns

def plot_time_series_with_relevance(data_batch: torch.Tensor, 
                                  heatmaps: torch.Tensor,
                                  channel_names: List[str] = ['X', 'Y', 'Z'],
                                  figsize: Tuple[int, int] = (12, 8)) -> List[plt.Figure]:
    """
    Plot time series data with relevance heatmaps
    
    Parameters:
    ----------
    data_batch: torch.Tensor - Shape [batch, 3, 2000] 
    heatmaps: torch.Tensor - Relevance values [batch, 2000]
    channel_names: List[str] - Names for the 3 channels
    
    Returns:
    --------
    List of matplotlib figures
    """
    figures = []
    
    for i in range(len(data_batch)):
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Plot each channel
        for ch in range(3):
            axes[ch].plot(data_batch[i, ch, :].cpu().numpy(), 
                         color=f'C{ch}', label=channel_names[ch])
            axes[ch].set_ylabel(f'{channel_names[ch]} Amplitude')
            axes[ch].legend()
            axes[ch].grid(True, alpha=0.3)
        
        # Plot relevance heatmap
        time_points = np.arange(len(heatmaps[i]))
        relevance = heatmaps[i].cpu().numpy()
        
        # Normalize relevance for color mapping
        rel_norm = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-8)
        
        axes[3].fill_between(time_points, 0, relevance, 
                           alpha=0.7, color='red', label='Relevance')
        axes[3].set_ylabel('Relevance')
        axes[3].set_xlabel('Time Steps')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.suptitle(f'Time Series Sample {i+1} with Relevance')
        plt.tight_layout()
        figures.append(fig)
    
    return figures

def plot_channel_importance(relevance_per_channel: torch.Tensor,
                          channel_names: List[str] = ['X', 'Y', 'Z']) -> plt.Figure:
    """
    Plot bar chart showing channel importance
    
    Parameters:
    ----------
    relevance_per_channel: torch.Tensor - Shape [3] or [batch, 3]
    """
    if len(relevance_per_channel.shape) > 1:
        rel_mean = relevance_per_channel.mean(dim=0)
        rel_std = relevance_per_channel.std(dim=0)
    else:
        rel_mean = relevance_per_channel
        rel_std = None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    bars = ax.bar(channel_names, rel_mean.cpu().numpy(), 
                  color=['blue', 'green', 'red'], alpha=0.7)
    
    if rel_std is not None:
        ax.errorbar(channel_names, rel_mean.cpu().numpy(), 
                   yerr=rel_std.cpu().numpy(), fmt='none', color='black')
    
    ax.set_ylabel('Average Relevance')
    ax.set_title('Channel Importance for Classification')
    ax.grid(True, alpha=0.3)
    
    return fig

def visualize_temporal_concepts(activations: torch.Tensor,
                              layer_name: str,
                              top_k: int = 5) -> plt.Figure:
    """
    Visualize temporal activation patterns for concepts
    
    Parameters:
    ----------
    activations: torch.Tensor - Shape [batch, channels, time]
    layer_name: str - Name of the layer
    top_k: int - Number of top channels to visualize
    """
    # Average across batch dimension
    avg_activations = activations.mean(dim=0)  # [channels, time]
    
    # Find top-k most active channels
    channel_importance = avg_activations.abs().sum(dim=1)
    top_channels = torch.topk(channel_importance, top_k)[1]
    
    fig, axes = plt.subplots(top_k, 1, figsize=(12, 2*top_k), sharex=True)
    if top_k == 1:
        axes = [axes]
    
    for i, ch_idx in enumerate(top_channels):
        axes[i].plot(avg_activations[ch_idx].cpu().numpy())
        axes[i].set_ylabel(f'Channel {ch_idx}')
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time Steps')
    plt.suptitle(f'Top-{top_k} Temporal Activation Patterns - {layer_name}')
    plt.tight_layout()
    
    return fig