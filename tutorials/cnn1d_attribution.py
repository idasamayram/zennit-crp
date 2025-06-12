import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.collections import LineCollection

# Add the parent directory to the path for imports
sys.path.append('../../zennit-crp')

# Import necessary modules from zennit and crp
from zennit.composites import EpsilonPlus
from zennit.canonizers import SequentialMergeBatchNorm
from crp.attribution import CondAttribution
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names, abs_norm
from crp.attribution import AttributionGraph
from crp.graph import trace_model_graph

# Import our custom concept for time series
from crp.time_series_concept import TimeSeriesConcept, ChannelTimeSeriesConcept

# Import relevance visualization tools
from crp.relevance_visualization import visualize_crp_heatmap

# Import your CNN1D model
from models.cnn1D_model import CNN1D_Wide, VibrationDataset


def load_model_and_sample(model_path, data_dir, device=None, sample_idx=42):
    """
    Load a trained CNN1D model and a sample from the dataset

    Parameters:
        model_path: Path to the saved model checkpoint
        data_dir: Directory containing the dataset
        device: Device to use (cpu or cuda)
        sample_idx: Index of the sample to use from the dataset

    Returns:
        model: Loaded CNN1D model
        sample: A batch of samples from the dataset
        label: True label for the sample
        is_synthetic: Whether the data is synthetic or real
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create and load the model
    model = CNN1D_Wide()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    model.to(device)

    try:
        # Try to load a sample from the test dataset
        print(f"Attempting to load dataset from: {data_dir}")

        # Modify VibrationDataset to bypass the assertion error
        class ModifiedVibrationDataset(VibrationDataset):
            def __init__(self, data_dir, augment_bad=False):
                super(VibrationDataset, self).__init__()
                self.data_dir = Path(data_dir)
                self.file_paths = []
                self.labels = []
                self.operations = []
                self.augment_bad = augment_bad
                self.file_groups = []

                for label, label_idx in zip(["good", "bad"], [0, 1]):
                    folder = self.data_dir / label
                    if folder.exists():
                        for file_name in folder.glob("*.h5"):
                            self.file_paths.append(file_name)
                            self.labels.append(label_idx)
                            # Extract operation from filename
                            operation = file_name.stem.split('_')[3] if len(
                                file_name.stem.split('_')) > 3 else 'unknown'
                            self.operations.append(operation)
                            # Extract file group
                            file_group = file_name.stem.rsplit('_window_', 1)[
                                0] if '_window_' in file_name.stem else file_name.stem
                            self.file_groups.append(file_group)

                self.labels = np.array(self.labels) if self.labels else np.array([])
                self.operations = np.array(self.operations) if self.operations else np.array([])
                self.file_groups = np.array(self.file_groups) if self.file_groups else np.array([])

                print(f"Found {len(self.file_paths)} files.")

        test_dataset = ModifiedVibrationDataset(data_dir)

        if len(test_dataset.file_paths) > 0:
            # Get a sample (choose a specific index if needed)
            sample_idx = min(sample_idx, len(test_dataset) - 1)  # Ensure index is valid
            sample, label = test_dataset[sample_idx]
            is_synthetic = False
        else:
            # Create synthetic data for testing
            print("Creating synthetic sample data")
            np.random.seed(42)  # For reproducibility
            # Generate a realistic-looking synthetic signal with frequency components
            t = np.linspace(0, 5, 2000)
            # Create 3 signals with different frequency components
            signal_x = 0.5 * np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 20 * t)
            signal_y = 0.7 * np.sin(2 * np.pi * 15 * t) + 0.2 * np.sin(2 * np.pi * 30 * t)
            signal_z = 0.6 * np.sin(2 * np.pi * 5 * t) + 0.4 * np.sin(2 * np.pi * 25 * t)

            # Add some noise
            signal_x += 0.1 * np.random.randn(2000)
            signal_y += 0.1 * np.random.randn(2000)
            signal_z += 0.1 * np.random.randn(2000)

            # Add a "fault pattern" to make this look like a "bad" sample
            # Add an impulse at a specific time
            impulse_loc = 1000
            impulse_width = 200
            impulse = np.exp(-0.01 * np.arange(-impulse_width / 2, impulse_width / 2) ** 2)
            signal_x[impulse_loc - impulse_width // 2:impulse_loc + impulse_width // 2] += 2 * impulse
            signal_y[impulse_loc - impulse_width // 2:impulse_loc + impulse_width // 2] += 1.5 * impulse

            # Combine into a tensor
            sample = torch.tensor(np.vstack([signal_x, signal_y, signal_z]), dtype=torch.float32)
            label = 1  # Set to "bad" class
            is_synthetic = True
            print("Created synthetic 'bad' vibration sample with a fault pattern")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using synthetic sample data instead")
        # Create a more realistic synthetic sample as a backup
        np.random.seed(42)
        t = np.linspace(0, 5, 2000)
        signal_x = 0.5 * np.sin(2 * np.pi * 10 * t)
        signal_y = 0.7 * np.sin(2 * np.pi * 15 * t)
        signal_z = 0.6 * np.sin(2 * np.pi * 5 * t)
        sample = torch.tensor(np.vstack([signal_x, signal_y, signal_z]), dtype=torch.float32)
        label = 0  # "Good" label
        is_synthetic = True

    # Add batch dimension and move to device
    sample = sample.unsqueeze(0).to(device)
    print(f"Sample shape: {sample.shape}, Label: {label}")

    print(f"✅ Model is structured as  \n {model}")

    return model, sample, label, is_synthetic


def visualize_time_series_attribution(data, attribution, title=None, figsize=(15, 10)):
    """
    Visualize time series data with attribution heatmap overlay
    """
    # Move data to CPU and convert to numpy
    if isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    if isinstance(attribution, torch.Tensor):
        attribution = attribution.cpu().detach().numpy()

    print(f"Data shape: {data.shape}")
    print(f"Attribution shape: {attribution.shape}")

    # Remove batch dimension if present
    if data.ndim == 3:
        data = data.squeeze(0)
    if attribution.ndim == 3:
        attribution = attribution.squeeze(0)

    print(f"After removing batch dimension - Data shape: {data.shape}")
    print(f"After removing batch dimension - Attribution shape: {attribution.shape}")

    # Debug: Check if channels have different attribution values
    print(f"X-Y axes identical: {np.allclose(attribution[0], attribution[1], atol=1e-5)}")
    print(f"Y-Z axes identical: {np.allclose(attribution[1], attribution[2], atol=1e-5)}")
    print(f"X-Z axes identical: {np.allclose(attribution[0], attribution[2], atol=1e-5)}")

    # Additional debug: Print stats for each channel
    for i in range(3):
        print(
            f"Axis {i} - min: {attribution[i].min():.6f}, max: {attribution[i].max():.6f}, mean: {attribution[i].mean():.6f}")
        # Print a sample of values to compare
        print(f"First 5 values: {attribution[i][:5]}")

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    axes_names = ["X", "Y", "Z"]

    # Plot each dimension separately
    for dim in range(3):
        ax = axes[dim]

        # Get dimension data
        dim_data = data[dim]
        dim_attr = attribution[dim]
        time_steps = np.arange(len(dim_data))

        # Print info about current dimension
        print(f"Plotting axis {dim} ({axes_names[dim]})")

        # Normalize attribution for coloring - per channel normalization
        # norm = plt.Normalize(vmin=dim_attr.min(), vmax=dim_attr.max())
        # cmap = plt.get_cmap('coolwarm')

        norm = plt.Normalize(vmin=-dim_attr.max(), vmax=dim_attr.max())
        cmap = plt.colormaps['bwr']


        # Create colored background rectangles based on attribution
        for t in range(len(time_steps) - 1):
            ax.axvspan(time_steps[t], time_steps[t + 1],
                       color=cmap(norm(dim_attr[t])), alpha=0.5)

        # Plot the signal as a black line on top
        ax.plot(time_steps, dim_data, color='black', linewidth=1.0, label="Signal")

        # Set plot limits and labels
        ax.set_xlim(0, len(dim_data))
        ax.set_ylim(np.min(dim_data) - 0.1, np.max(dim_data) + 0.1)
        ax.set_xlabel('Time')
        ax.set_ylabel(f'{axes_names[dim]} Axis')

        # Add a color bar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, pad=0.01, label='Attribution Value')

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    return fig


# Custom class to handle heatmap generation for time series
class TimeSeriesCondAttribution(CondAttribution):
    """
    Extension of CondAttribution with custom heatmap modifier for time series data
    """

    def heatmap_modifier(self, data, on_device=None):
        """
        Custom heatmap modifier that preserves channel structure for time series data
        """
        heatmap = data.grad.detach()

        # Ensure we return a tensor with shape (channels, time_steps)
        # Don't squeeze out the channel dimension
        if heatmap.dim() >= 3:  # (batch, channels, time_steps)
            heatmap = heatmap[0]  # Remove batch dim, keep channels

        heatmap = heatmap.to(on_device) if on_device else heatmap
        return heatmap


def apply_crp_to_cnn1d(model, sample, target_class=None, layer_name=None, concept_idx=None):
    """
    Apply Concept Relevance Propagation to a CNN1D model

    Parameters:
        model: The CNN1D model
        sample: Input time series data (batch_size, 3, 2000)
        target_class: Target class index for the attribution (if None, uses predicted class)
        layer_name: Name of the layer to focus on (if None, generates heatmap for model output)
        concept_idx: Concept/channel index to focus on (if None, no specific channel is selected)

    Returns:
        attribution: Attribution result
    """
    # Set up the device
    device = next(model.parameters()).device

    # Enable gradient calculation for the input
    sample.requires_grad = True

    # Get layer names of convolutional and linear layers
    layer_names = get_layer_names(model, [torch.nn.Conv1d, torch.nn.Linear])

    # Set up the composite using EpsilonPlusFlat with canonizers for GroupNorm
    composite = EpsilonPlus()

    # Initialize our custom concept class for time series data
    cc = TimeSeriesConcept()

    # Set up our custom attribution object
    attribution = TimeSeriesCondAttribution(model, no_param_grad=True)

    # Determine target class if not provided
    if target_class is None:
        with torch.no_grad():
            output = model(sample)
            target_class = output.argmax(dim=1).item()
            print(f"Using predicted class: {target_class}")

    # Build conditions based on inputs
    if layer_name is not None and concept_idx is not None:
        # Condition on specific layer and concept
        conditions = [{"y": [target_class], layer_name: [concept_idx]}]
    else:
        # Condition only on the output class
        conditions = [{"y": [target_class]}]

    # Compute attribution
    attr = attribution(sample, conditions, composite, record_layer=layer_names)

    # Debug the heatmap shape
    print(f"Raw heatmap shape: {attr.heatmap.shape}")

    return attr


def find_important_concepts(model, sample, target_class, layer_name):
    """
    Find the most important concepts/channels in a specific layer

    Parameters:
        model: The CNN1D model
        sample: Input time series data
        target_class: Target class for attribution
        layer_name: Name of the layer to analyze

    Returns:
        concept_ids: Indices of the most important concepts
        relevance_values: Corresponding relevance values
    """
    # Set up attribution
    device = next(model.parameters()).device
    sample.requires_grad = True

    # Use our custom concept for time series
    cc = TimeSeriesConcept()
    composite = EpsilonPlus()
    attribution = TimeSeriesCondAttribution(model, no_param_grad=True)

    # Apply attribution with conditions on target class only
    conditions = [{"y": [target_class]}]
    attr = attribution(sample, conditions, composite, record_layer=[layer_name])

    # Get relevance for each concept in the layer
    rel_c = cc.attribute(attr.relevances[layer_name], abs_norm=True)

    # Get the top concepts (or all concepts if fewer than 6)
    num_concepts = min(10, rel_c.shape[1])
    rel_values, concept_ids = torch.topk(rel_c[0], num_concepts)

    return concept_ids, rel_values


def analyze_time_series_with_crp(model_path, data_dir, target_class=None, top_n=5, sample_idx=42):
    """
    Analyze a time series model using Concept Relevance Propagation

    Parameters:
        model_path: Path to the saved model
        data_dir: Path to the data directory
        target_class: Target class for attribution (if None, uses predicted class)
        top_n: Number of top concepts to analyze
        sample_idx: Index of the sample to use for analysis
    """
    # Load model and sample
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, sample, true_label, is_synthetic = load_model_and_sample(model_path, data_dir, device, sample_idx)

    print(f"Loaded model and sample. True label: {true_label} ({'Synthetic' if is_synthetic else 'Real'} data)")
    print(f"Sample shape: {sample.shape}")

    # If target class not specified, use predicted class
    if target_class is None:
        with torch.no_grad():
            output = model(sample)
            target_class = output.argmax(dim=1).item()
        print(
            f"Using predicted class: {target_class} (confidence: {torch.softmax(output, dim=1)[0, target_class].item():.4f})")

    # Get layer names
    layer_names = get_layer_names(model, [torch.nn.Conv1d, torch.nn.Linear])
    print(f"Found layers: {layer_names}")

    # 1. Overall attribution for classification (what parts of the signal are important)
    print("Computing overall attribution...")
    attr = apply_crp_to_cnn1d(model, sample, target_class)

    # Visualize the overall attribution using our custom function
    fig = visualize_time_series_attribution(
        sample,
        attr.heatmap,
        title=f"Attribution for Class {target_class}"
    )
    # fig.savefig("overall_attribution.png")
    fig.savefig("./output/overall_attribution.png")

    plt.close(fig)

    # Try to also use the visualization from relevance_visualization.py
    try:
        # Make sure shapes match correctly
        sample_for_viz = sample.clone()  # Clone to avoid modifying the original
        if isinstance(sample_for_viz, torch.Tensor):
            sample_for_viz = sample_for_viz.cpu().detach().numpy()

        heatmap_for_viz = attr.heatmap.clone()  # Clone to avoid modifying the original
        if isinstance(heatmap_for_viz, torch.Tensor):
            heatmap_for_viz = heatmap_for_viz.cpu().detach().numpy()

        # If both have batch dimensions, remove them
        if sample_for_viz.ndim == 3:
            sample_for_viz = sample_for_viz.squeeze(0)
        if heatmap_for_viz.ndim == 3:
            heatmap_for_viz = heatmap_for_viz.squeeze(0)

        # If heatmap is (1, 2000), calculate per-channel attributions
        if heatmap_for_viz.shape[0] == 1 and sample_for_viz.shape[0] == 3:
            print("Creating per-channel heatmaps for visualization...")
            # Create per-channel heatmaps by multiplying with normalized data magnitude
            channel_heatmaps = np.zeros_like(sample_for_viz)
            for i in range(sample_for_viz.shape[0]):
                channel_norm = np.abs(sample_for_viz[i]) / (np.max(np.abs(sample_for_viz[i])) + 1e-10)
                channel_heatmaps[i] = heatmap_for_viz[0] * channel_norm
            heatmap_for_viz = channel_heatmaps

        print(f"Final sample shape for viz: {sample_for_viz.shape}")
        print(f"Final heatmap shape for viz: {heatmap_for_viz.shape}")

        visualize_crp_heatmap(
            sample_for_viz,
            heatmap_for_viz,
            label=true_label,
            method_name="CRP"
        )
    except Exception as e:
        print(f"Could not use visualize_crp_heatmap: {e}")

    # 2. Find important concepts in convolutional layers
    for layer_name in layer_names:
        if 'conv' in layer_name:  # Focus on convolutional layers
            print(f"\nAnalyzing layer {layer_name}...")
            concept_ids, relevance_values = find_important_concepts(model, sample, target_class, layer_name)

            print(f"Top {len(concept_ids)} concepts in {layer_name}:")
            for i, (concept_id, rel_val) in enumerate(zip(concept_ids.tolist(), relevance_values.tolist())):
                print(f"  Concept {concept_id}: {rel_val * 100:.2f}%")

            # 3. Generate conditional attributions for top concepts
            print(f"Generating conditional attributions for top concepts in {layer_name}...")
            for i, concept_id in enumerate(concept_ids[:min(top_n, len(concept_ids))]):
                try:
                    attr = apply_crp_to_cnn1d(model, sample, target_class, layer_name, concept_id)

                    # Visualize the conditional attribution
                    fig = visualize_time_series_attribution(
                        sample,
                        attr.heatmap,
                        title=f"Attribution for Concept {concept_id} in {layer_name}"
                    )
                    # fig.savefig(f"{layer_name}_concept_{concept_id}_attribution.png")
                    fig.savefig(f"./output/{layer_name}_concept_{concept_id}_attribution.png")

                    plt.close(fig)
                except Exception as e:
                    print(f"Error visualizing concept {concept_id}: {e}")

    # 4. Optional: Analyze concept decomposition using AttributionGraph
    try:
        print("\nTracing model graph for concept decomposition...")
        graph = trace_model_graph(model, sample, layer_names)

        # Set up AttributionGraph
        cc = TimeSeriesConcept()  # Use our custom concept
        layer_map = {name: cc for name in layer_names}
        attribution = TimeSeriesCondAttribution(model, no_param_grad=True)  # Use our custom attribution
        composite = EpsilonPlus()

        attgraph = AttributionGraph(attribution, graph, layer_map)

        # Choose a concept to decompose (using the first important concept from the last conv layer)
        conv_layers = [name for name in layer_names if 'conv' in name]
        if conv_layers:
            last_conv_layer = conv_layers[-1]
            concept_ids, _ = find_important_concepts(model, sample, target_class, last_conv_layer)
            concept_to_decompose = concept_ids[0].item()

            print(f"Decomposing concept {concept_to_decompose} in {last_conv_layer}...")
            nodes, connections = attgraph(sample, composite, concept_to_decompose,
                                          last_conv_layer, target_class, width=[3, 2], abs_norm=True)

            print("Concept Decomposition:")
            print("Nodes:", nodes)
            print("Connections:", connections)
    except Exception as e:
        print(f"Could not perform concept decomposition: {e}")


# For channel-specific concepts
def apply_channel_specific_crp(model, sample, target_class, channel_idx=0):
    """
    Apply CRP with focus on a specific channel of the time series

    Parameters:
        model: The CNN1D model
        sample: Input time series data (batch_size, 3, 2000)
        target_class: Target class index
        channel_idx: Channel index to focus on (0, 1, or 2 for X, Y, Z)

    Returns:
        attribution: Attribution result
    """
    device = next(model.parameters()).device
    sample.requires_grad = True

    # Create custom input mask to focus on one channel
    input_mask = torch.zeros_like(sample)
    input_mask[:, channel_idx, :] = 1.0

    # Apply the mask to the input
    masked_sample = sample.clone()
    masked_sample.retain_grad()

    # Get layer names
    layer_names = get_layer_names(model, [torch.nn.Conv1d, torch.nn.Linear])
    composite = EpsilonPlus()

    # Use our custom attribution and concept classes
    cc = ChannelTimeSeriesConcept()
    attribution = TimeSeriesCondAttribution(model, no_param_grad=True)

    # Create a condition that focuses on the specific channel
    channel_condition = {channel_idx: list(range(sample.shape[2]))}  # All time points in the channel
    conditions = [{"y": [target_class], "input": channel_condition}]

    # Compute attribution with focus on one channel
    attr = attribution(masked_sample, conditions, composite, record_layer=layer_names)

    return attr


if __name__ == "__main__":
    # Specify paths
    model_path = "../models/cnn1d_model.ckpt"

    # Try different possible locations for the dataset
    data_dirs = ["../time_data"]  # Original path]
    # data_dirs = "E:/Thesis/Datasets/CNC/data/final/new_selection/normalized_windowed_downsampled_data"

    # Try to find a valid data directory
    data_dir = None
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            data_dir = dir_path
            print(f"Found dataset at: {data_dir}")
            break

    if data_dir is None:
        print(f"Warning: Could not find dataset directory. Using synthetic data.")
        data_dir = data_dirs[0]  # Use first path anyway for the function call

    # Run the analysis
    analyze_time_series_with_crp(model_path, data_dir)