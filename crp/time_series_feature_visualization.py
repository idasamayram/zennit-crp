from crp.visualization import FeatureVisualization
from time_series_visualization import plot_time_series_with_relevance
import torch

class TimeSeriesFeatureVisualization(FeatureVisualization):
    """
    Adapted FeatureVisualization for time series data
    """
    
    def __init__(self, attribution, dataset, layer_map, preprocess_fn=None, **kwargs):
        # Remove cache parameter for now - we'll implement a simple version
        super().__init__(attribution, dataset, layer_map, preprocess_fn, **kwargs)
    
    def get_data_sample(self, index, preprocessing=True):
        """
        Get time series sample from your VibrationDataset
        """
        data, target = self.dataset[index]
        data = data.to(self.device).unsqueeze(0)
        
        if preprocessing and callable(self.preprocess_fn):
            data = self.preprocess_fn(data)
        
        data.requires_grad = True
        return data, target
    
    def multitarget_to_single(self, target):
        """
        Convert multi-label to single label (if needed)
        For binary classification, just return the target
        """
        if isinstance(target, (int, torch.Tensor)):
            return [target] if not isinstance(target, list) else target
        return [target]

# Usage example for your model:
def create_time_series_crp_analysis(model, dataset, device):
    """
    Create CRP analysis setup for your CNN1D model
    """
    from crp.attribution import CondAttribution
    from crp.helper import get_layer_names
    from time_series_concepts import TimeSeriesConcept
    
    # Get layer names for 1D CNN layers
    layer_names = get_layer_names(model, [torch.nn.Conv1d, torch.nn.Linear])
    
    # Create layer mapping with TimeSeriesConcept
    layer_map = {name: TimeSeriesConcept() for name in layer_names}
    
    # Create attribution object
    attribution = CondAttribution(model, device)
    
    # Create feature visualization
    fv = TimeSeriesFeatureVisualization(
        attribution=attribution,
        dataset=dataset,
        layer_map=layer_map,
        preprocess_fn=None,  # Add your preprocessing if needed
        device=device
    )
    
    return fv, attribution, layer_map