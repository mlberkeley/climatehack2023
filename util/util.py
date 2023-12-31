import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch


def to_np(a):
    return a.detach().cpu().numpy()


def visualize_example(pv_feature, pv_target, pred, hrv_feature):
    pv_feature = to_np(pv_feature)
    pv_target = to_np(pv_target)
    pred = to_np(pred)
    hrv_feature = to_np(hrv_feature * 255).astype(np.uint8)
    hrv_feature = hrv_feature.reshape(12, 1, 128, 128).repeat(3, axis=1)

    fig, ax = plt.subplots()

    ax.plot(np.arange(0, 12), pv_feature, color='black', label="features")
    ax.plot(np.arange(12, 60), pv_target, color='green', label="target")
    ax.plot(np.arange(12, 60), pred, color='red', label="prediction")
    ax.plot([11,12], [pv_feature[-1], pv_target[0]], color='black')

    ax.legend()
    plt.tight_layout()

    return wandb.Image(fig), wandb.Video(hrv_feature, fps=1, format="gif")

