import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def s_layer(input_img, filters):
    feature_maps = []
    for filt in filters:
        conv = convolve2d(input_img, filt, mode='valid')
        conv = np.maximum(conv, 0)  # ReLU
        feature_maps.append(conv)
    return feature_maps

def c_layer(feature_maps, pool_size=2):
    pooled_maps = []
    for fmap in feature_maps:
        h, w = fmap.shape
        pooled = np.zeros((h // pool_size, w // pool_size))
        for i in range(0, h, pool_size):
            for j in range(0, w, pool_size):
                window = fmap[i:i+pool_size, j:j+pool_size]
                pooled[i // pool_size, j // pool_size] = np.max(window)
        pooled_maps.append(pooled)
    return pooled_maps

def visualize_neocognitron_layers(img, filters, s_features, c_features):
    fig, axs = plt.subplots(2, len(filters) + 1, figsize=(12, 6))
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title("Input")
    axs[0, 0].axis('off')

    for i in range(len(filters)):
        axs[0, i+1].imshow(s_features[i], cmap='gray')
        axs[0, i+1].set_title(f"S-layer {i+1}")
        axs[0, i+1].axis('off')

        axs[1, i+1].imshow(c_features[i], cmap='gray')
        axs[1, i+1].set_title(f"C-layer {i+1}")
        axs[1, i+1].axis('off')

    axs[1, 0].axis('off')
    plt.tight_layout()
    plt.show()
