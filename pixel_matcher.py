import copy
import time
import math
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class PixelMatcher:
    def __init__(self, imgs, features, img_size=640, reference_pixel_keypoints=None,
                 target_mask=None, window_size=1, patch_mode='single',
                 selection_mode='argmax', softargmax_temp=0.05, threshold_ratio=0.5,
                 offset_y=80):
        self.ft = features
        self.imgs = imgs
        self.img_size = img_size
        self.reference_pixel_keypoints = reference_pixel_keypoints
        self.target_mask = target_mask
        self.window_size = window_size
        self.patch_mode = patch_mode
        self.selection_mode = selection_mode
        self.softargmax_temp = softargmax_temp
        self.threshold_ratio = threshold_ratio
        self.offset_y = offset_y

        assert patch_mode in ['single', 'avg', 'patch']
        assert selection_mode in ['argmax', 'softargmax', 'threshold_centroid', 'gaussian_centroid']
        assert window_size % 2 == 1

        self.src_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(self.ft[0].unsqueeze(0))
        self.trg_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(self.ft[1].unsqueeze(0))

        if target_mask is not None:
            assert target_mask.shape == (img_size, img_size)
            mask_tensor = torch.tensor(target_mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.trg_ft *= mask_tensor

    def _extract_patch(self, ft_map, x, y):
        half = self.window_size // 2
        x = np.clip(x, half, self.img_size - half - 1)
        y = np.clip(y, half, self.img_size - half - 1)
        return ft_map[:, x - half: x + half + 1, y - half: y + half + 1]

    def _compute_cos_map(self, src_ft, trg_ft, src_x, src_y):
        cos = nn.CosineSimilarity(dim=1)
        C = src_ft.size(0)

        if self.patch_mode == 'single':
            src_vec = src_ft[:, src_x, src_y].view(1, C, 1, 1)
            cos_map = cos(src_vec, trg_ft.unsqueeze(0)).squeeze(0).detach().cpu().numpy()

        elif self.patch_mode == 'avg':
            patch = self._extract_patch(src_ft, src_x, src_y)
            src_vec = patch.mean(dim=(1, 2), keepdim=True).view(1, C, 1, 1)
            cos_map = cos(src_vec, trg_ft.unsqueeze(0)).squeeze(0).detach().cpu().numpy()

        elif self.patch_mode == 'patch':
            patch = self._extract_patch(src_ft, src_x, src_y)
            C, H, W = trg_ft.shape
            ws = self.window_size
            half = ws // 2
            src_patch = F.normalize(patch, dim=0).unsqueeze(1)
            trg_map = F.normalize(trg_ft, dim=0).unsqueeze(0)
            sim_map = F.conv2d(trg_map, weight=src_patch, padding=half, groups=C)
            cos_map = sim_map.squeeze(0).sum(0).detach().cpu().numpy()

        else:
            raise ValueError(f"Invalid patch_mode: {self.patch_mode}")

        return cos_map

    def _select_pixel(self, cos_map):
        if self.selection_mode == 'argmax':
            return np.unravel_index(np.argmax(cos_map), cos_map.shape)

        elif self.selection_mode == 'softargmax':
            norm_map = cos_map - np.max(cos_map)
            prob_map = np.exp(norm_map / self.softargmax_temp)
            prob_map /= np.sum(prob_map)
            yx = np.indices(cos_map.shape)
            y = (yx[0] * prob_map).sum()
            x = (yx[1] * prob_map).sum()
            return int(round(y)), int(round(x))

        elif self.selection_mode == 'threshold_centroid':
            threshold = self.threshold_ratio * np.max(cos_map)
            mask = cos_map >= threshold
            if np.sum(mask) == 0:
                return np.unravel_index(np.argmax(cos_map), cos_map.shape)
            yx = np.argwhere(mask)
            return tuple(np.round(yx.mean(axis=0)).astype(int))

        elif self.selection_mode == 'gaussian_centroid':
            threshold = self.threshold_ratio
            weights = np.where(cos_map >= threshold, cos_map, 0)
            total_weight = np.sum(weights)
            if total_weight == 0:
                return np.unravel_index(np.argmax(cos_map), cos_map.shape)
            yx = np.indices(cos_map.shape)
            y = (yx[0] * weights).sum() / total_weight
            x = (yx[1] * weights).sum() / total_weight
            return int(round(y)), int(round(x))

        else:
            raise ValueError(f"Invalid selection_mode: {self.selection_mode}")

    def _match_single_point(self, src_x, src_y):
        cos_map = self._compute_cos_map(self.src_ft[0], self.trg_ft[0], src_x, src_y)
        if self.target_mask is not None:
            cos_map = np.where(self.target_mask, cos_map, -np.inf)
        best_y, best_x = self._select_pixel(cos_map)
        return best_y, best_x, cos_map

    def _plot_side_by_side(self, axes, x_click, y_click, best_x, best_y, cos_map, alpha, scatter_size, source_title, target_title, mark_style='o'):
        axes[0].clear()
        axes[0].imshow(self.imgs[0])
        axes[0].axis('off')
        axes[0].scatter(x_click, y_click, c='r', s=scatter_size, marker=mark_style)
        axes[0].set_title(source_title)

        cosmap = copy.deepcopy(cos_map)
        cosmap[cosmap == -np.inf] = 0
        heatmap = (cosmap - np.min(cosmap)) / (np.max(cosmap) - np.min(cosmap))
        sharpness = 10
        heatmap = 1 / (1+np.exp(-sharpness*(heatmap-0.8)))

        axes[1].clear()
        axes[1].imshow(self.imgs[1])
        axes[1].imshow(255 * heatmap, alpha=alpha, cmap='viridis')
        axes[1].axis('off')
        axes[1].scatter(best_x, best_y, c='r', s=scatter_size, marker=mark_style)
        axes[1].set_title(target_title)

    def _click_and_match_one(self, fig_size=5, alpha=0.75, scatter_size=70, keypoint_name=None):
        fig, axes = plt.subplots(1, 2, figsize=(fig_size * 2, fig_size))
        plt.tight_layout()
        for i in range(2):
            axes[i].imshow(self.imgs[i])
            axes[i].axis('off')
        if keypoint_name:
            axes[0].set_title(f"Click for {keypoint_name}")
            axes[1].set_title("Predicted Correspondence")
        else:
            axes[0].set_title("source image")
            axes[1].set_title("target image")

        picked = {}

        def onclick(event):
            if event.inaxes == axes[0]:
                with torch.no_grad():
                    x_click = int(np.clip(round(event.xdata), 0, self.img_size - 1))
                    y_click = int(np.clip(round(event.ydata), 0, self.img_size - 1))

                    best_y, best_x, cos_map = self._match_single_point(y_click, x_click)

                    source_title = f"Selected {keypoint_name}" if keypoint_name else "source image"
                    target_title = f"Predicted for {keypoint_name}" if keypoint_name else "target image"
                    mark_style = 'x' if keypoint_name else 'o'

                    self._plot_side_by_side(axes, x_click, y_click, best_x, best_y, cos_map,
                                            alpha, scatter_size, source_title, target_title, mark_style)

                    fig.canvas.draw()
                    del cos_map
                    gc.collect()

                    picked['x'] = x_click
                    picked['y'] = y_click
                    picked['best_x'] = best_x
                    picked['best_y'] = best_y

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        if picked:
            return picked['y'], picked['x'], picked['best_y'], picked['best_x']
        else:
            return None

    def find_corresponding_pixel_keypoints_interactive(self, fig_size=5, alpha=0.75, scatter_size=70):
        if self.reference_pixel_keypoints is None:
            raise ValueError("Pixel keypoints dictionary is required (for keys).")

        selected_pixel_keypoints = {}
        matched_pixel_keypoints = {}

        for keypoint_name in self.reference_pixel_keypoints.keys():
            result = self._click_and_match_one(fig_size, alpha, scatter_size, keypoint_name)
            if result:
                selected_y, selected_x, matched_y, matched_x = result
                selected_pixel_keypoints[keypoint_name] = (selected_y - self.offset_y, selected_x)
                matched_pixel_keypoints[keypoint_name] = (matched_y - self.offset_y, matched_x)

        self.selected_pixel_keypoints = selected_pixel_keypoints.copy()
        self.matched_pixel_keypoints = matched_pixel_keypoints.copy()
        return selected_pixel_keypoints, matched_pixel_keypoints

    def find_corresponding_pixel_keypoints(self, r_n="Source Image", t_n="Target Image", fig_size=5, alpha=0.75, scatter_size=70, visualize=True, save=True):
        if self.reference_pixel_keypoints is None:
            raise ValueError("No pixel_keypoints provided.")

        matched_pixel_keypoints = {}

        if visualize or save:
            fig, axes = plt.subplots(1, 2, figsize=(fig_size * 2, fig_size))
            for i in range(2):
                axes[i].imshow(self.imgs[i])
                axes[i].axis('off')
            axes[0].set_title(r_n)
            axes[1].set_title(t_n)
            plt.tight_layout()

        for keypoint_name, (row, col) in self.reference_pixel_keypoints.items():
            selected_x = row + self.offset_y
            selected_y = col
            best_y, best_x, _ = self._match_single_point(selected_x, selected_y)
            matched_pixel_keypoints[keypoint_name] = (best_y - self.offset_y, best_x)

        if visualize or save:
            for _, xy in self.reference_pixel_keypoints.items():
                axes[0].scatter(xy[1], xy[0] + self.offset_y, s=scatter_size)
            for _, xy in matched_pixel_keypoints.items():
                axes[1].scatter(xy[1], xy[0] + self.offset_y, s=scatter_size)
            if save:
                plt.savefig(f"experiment_results/{r_n}_{t_n}.png")
            if visualize:
                plt.show()
            else:
                plt.close(fig)

        self.matched_pixel_keypoints = matched_pixel_keypoints.copy()
        return matched_pixel_keypoints
'''
    def visualize_matches(self, selected=False, save=True, visualize=True):
        fig, axes = plt.subplots(1, 2, figsize=(fig_size * 2, fig_size))
        for i in range(2):
            axes[i].imshow(self.imgs[i])
            axes[i].axis('off')
        axes[0].set_title("Source Image")
        axes[1].set_title("Target Image")
        plt.tight_layout()
        for _, xy in (self.reference_pixel_keypoints.items() if not selected else self.selected_pixel_keypoints.items()):
            axes[0].scatter(xy[1], xy[0] + self.offset_y, s=scatter_size)
        for _, xy in matched_pixel_keypoints.items():
            axes[1].scatter(xy[1], xy[0] + self.offset_y, s=scatter_size)
        if save:
            plt.savefig(f"experiment_results/{time.time_ns}.png")
        if visualize:
            plt.show()
        else:
            plt.close(fig)
'''
