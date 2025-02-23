import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Optional
import os
from skimage import io
from scipy.ndimage import gaussian_filter
from archs.unimedxseg import UniMedXSeg
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gc


class SegmentationCAM:
    def __init__(self, model: nn.Module, target_layers: List[str], smooth_sigma: float = 0.05,
                 threshold: float = 0.05, output_dir='original_cams'):
        self.model = model
        self.target_layers = target_layers
        self.smooth_sigma = smooth_sigma
        self.threshold = threshold
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self.handles = []  # 存储所有hook句柄
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.cam_counter = 0
        self._register_hooks()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """清理所有资源"""
        # 清理hook句柄
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

        # 清理存储的激活值和梯度
        self.activations.clear()
        self.gradients.clear()

        # 清理GPU缓存
        torch.cuda.empty_cache()
        gc.collect()

    def _register_hooks(self):
        def save_activation(name: str):
            def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
                self.activations[name] = output.detach()

            return hook

        def save_gradient(name: str):
            def hook(module: nn.Module, grad_input: Tuple[torch.Tensor], grad_output: Tuple[torch.Tensor]):
                self.gradients[name] = grad_output[0].detach()

            return hook

        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.handles.append(module.register_forward_hook(save_activation(name)))
                self.handles.append(module.register_full_backward_hook(save_gradient(name)))

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.max() + 1e-8)

    def get_segmentation_cam(self, x: torch.Tensor) -> Tuple[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], torch.Tensor]:
        try:
            x = self._normalize_input(x)
            output = self.model(x)
            segmentation = output
            seg_cams = self._compute_cam_for_target(segmentation)
            return seg_cams, segmentation
        finally:
            # 清理中间变量
            self.activations.clear()
            self.gradients.clear()
            torch.cuda.empty_cache()

    def _compute_cam_for_target(self, target: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        try:
            self.model.zero_grad()
            target.sum().backward(retain_graph=False)
            positive_cams, negative_cams = self._compute_cam()
            return positive_cams, negative_cams
        finally:
            torch.cuda.empty_cache()

    def _compute_cam(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        positive_cams, negative_cams = {}, {}
        try:
            for layer_name in self.target_layers:
                if layer_name not in self.activations or layer_name not in self.gradients:
                    continue

                activations = self.activations[layer_name]
                grads = self.gradients[layer_name]

                positive_cams[layer_name] = self._compute_single_cam(activations, F.relu(grads),
                                                                     f"positive_{layer_name}")
                negative_cams[layer_name] = self._compute_single_cam(activations, F.relu(-grads),
                                                                     f"negative_{layer_name}")

            self.cam_counter += 1
            return positive_cams, negative_cams
        finally:
            # 确保计算完CAM后清理中间变量
            self.activations.clear()
            self.gradients.clear()

    def _compute_single_cam(self, activations: torch.Tensor, grads: torch.Tensor, cam_type: str) -> torch.Tensor:
        try:
            cam = torch.sum(grads * activations, dim=1, keepdim=True)

            # 检查当前特征图的尺寸
            current_size = cam.shape[-2:]
            target_size = (256, 256)

            # 只有当特征图小于目标尺寸时才进行插值
            if current_size[0] < target_size[0] or current_size[1] < target_size[1]:
                cam = F.interpolate(cam, size=target_size, mode='bilinear', align_corners=False)
            elif current_size[0] > target_size[0] or current_size[1] > target_size[1]:
                # 如果特征图大于目标尺寸，则下采样到目标尺寸
                cam = F.interpolate(cam, size=target_size, mode='bilinear', align_corners=False)

            cam = self._normalize(cam)
            cam = self._apply_smoothing(cam)
            cam = self._apply_thresholding(cam)
            return cam
        except Exception as e:
            print(f"Error in computing CAM for {cam_type}: {str(e)}")
            return torch.zeros((1, 1, 256, 256), device=activations.device)
        # try:
        #     cam = torch.sum(grads * activations, dim=1, keepdim=True)
        #     cam = F.interpolate(cam, size=(256, 256), mode='bilinear', align_corners=False)
        #     cam = self._normalize(cam)
        #     cam = self._apply_smoothing(cam)
        #     cam = self._apply_thresholding(cam)
        #     return cam
        # except Exception as e:
        #     print(f"Error in computing CAM for {cam_type}: {str(e)}")
        #     return torch.zeros((1, 1, 256, 256), device=activations.device)

    @staticmethod
    def _normalize(cam: torch.Tensor) -> torch.Tensor:
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    def _apply_smoothing(self, cam: torch.Tensor) -> torch.Tensor:
        cam_np = cam.detach().cpu().numpy()
        smoothed_cam = np.stack([gaussian_filter(c[0], sigma=self.smooth_sigma) for c in cam_np])
        return torch.from_numpy(smoothed_cam).unsqueeze(1).to(cam.device)

    def _apply_thresholding(self, cam: torch.Tensor) -> torch.Tensor:
        return F.threshold(cam, self.threshold, 0)


def apply_segmentation_cam(model: nn.Module, input_image: np.ndarray, target_layers: List[str]) -> Tuple[
    Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], torch.Tensor]:
    with SegmentationCAM(model, target_layers) as cam_model:
        # Handle both RGB and grayscale inputs
        if len(input_image.shape) == 2:  # 灰度图像
            input_tensor = torch.from_numpy(input_image.astype(np.float32)).unsqueeze(0).expand(3, -1, -1)  # 转换为RGB
        else:  # RGB图像
            input_tensor = torch.from_numpy(input_image.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0)

        # 确保输入张量的形状为 [1, 3, H, W]
        input_tensor = input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor

        input_tensor = input_tensor.to(next(model.parameters()).device)
        return cam_model.get_segmentation_cam(input_tensor)
    # with SegmentationCAM(model, target_layers) as cam_model:
    #     # Handle both RGB and grayscale inputs
    #     if len(input_image.shape) == 2:
    #         input_tensor = torch.from_numpy(input_image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    #     else:  # RGB image
    #         input_tensor = torch.from_numpy(input_image.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0)
    #
    #     input_tensor = input_tensor.to(next(model.parameters()).device)
    #     return cam_model.get_segmentation_cam(input_tensor)


def visualize_and_save_layers(input_image: np.ndarray,
                              seg_cams: Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
                              segmentation: torch.Tensor,
                              save_dir: str,
                              filename_prefix: str = "sample"):
    try:
        os.makedirs(save_dir, exist_ok=True)
        individual_activations_dir = os.path.join(save_dir, "individual_activations")
        os.makedirs(individual_activations_dir, exist_ok=True)

        positive_cams, negative_cams = seg_cams

        def enhance_cam(cam: np.ndarray, percentile: float = 97) -> np.ndarray:
            threshold = np.percentile(cam, percentile)
            cam = np.clip(cam, 0, threshold)
            return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        def visualize_overview(cams_dict: Dict[str, torch.Tensor], title_prefix: str, filename_suffix: str,
                               original_image: np.ndarray):
            try:
                n_layers = len(cams_dict)
                n_cols = 4
                n_rows = (n_layers + n_cols - 1) // n_cols

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
                if n_rows == 1:
                    axes = axes.reshape(1, -1)

                for i, (layer_name, cam) in enumerate(cams_dict.items()):
                    row, col = i // n_cols, i % n_cols
                    ax = axes[row, col]

                    cam_np = cam.detach().cpu().numpy().squeeze()
                    enhanced_cam = enhance_cam(cam_np)

                    if len(original_image.shape) == 3:
                        display_img = original_image.mean(axis=2)
                    else:
                        display_img = original_image
                    # 使用cv2或skimage调整原图尺寸
                    from skimage.transform import resize
                    display_img = resize(display_img, (256, 256), preserve_range=True, anti_aliasing=True)

                    ax.imshow(display_img, cmap='gray')
                    im = ax.imshow(enhanced_cam, cmap='jet', alpha=0.6)
                    ax.set_title(f'{title_prefix} - {layer_name}')
                    ax.axis('off')

                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)

                for i in range(len(cams_dict), n_rows * n_cols):
                    row, col = i // n_cols, i % n_cols
                    axes[row, col].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{filename_prefix}_{filename_suffix}_overview.png"),
                            dpi=300, bbox_inches='tight')
            finally:
                plt.close()

        def visualize_single_layer_cam(cam: torch.Tensor, title: str, filename: str, original_image: np.ndarray):
            try:
                plt.figure(figsize=(12, 5))

                if len(original_image.shape) == 3:
                    display_img = original_image.mean(axis=2)
                else:
                    display_img = original_image

                # 调整原图尺寸为256x256
                from skimage.transform import resize
                display_img = resize(display_img, (256, 256), preserve_range=True, anti_aliasing=True)

                cam_np = cam.detach().cpu().numpy().squeeze()
                enhanced_cam = enhance_cam(cam_np)

                plt.imshow(display_img, cmap='gray')
                im = plt.imshow(enhanced_cam, cmap='jet', alpha=0.6)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(individual_activations_dir, filename), dpi=300, bbox_inches='tight')
            finally:
                plt.close()

        visualize_overview(positive_cams, 'Positive CAM', 'positive', input_image)
        visualize_overview(negative_cams, 'Negative CAM', 'negative', input_image)

        for cams_dict, prefix in [(positive_cams, 'positive'),
                                  (negative_cams, 'negative')]:
            for layer_name, cam in cams_dict.items():
                title = f'{prefix.capitalize()} CAM - {layer_name}'
                filename = f"{filename_prefix}_{prefix}_{layer_name}.png"
                visualize_single_layer_cam(cam, title, filename, input_image)

        print(f"Saved visualizations to {save_dir}")
    finally:
        plt.close('all')
        gc.collect()


def process_image(model: nn.Module, image_path: str, target_layers: List[str], save_directory: str):
    try:
        image = io.imread(image_path)
        seg_cams, segmentation = apply_segmentation_cam(model, image, target_layers)

        image_filename = os.path.splitext(os.path.basename(image_path))[0]
        image_save_dir = os.path.join(save_directory, image_filename)

        visualize_and_save_layers(
            image, seg_cams, segmentation, image_save_dir,
            filename_prefix=image_filename
        )
    finally:
        plt.close('all')
        torch.cuda.empty_cache()
        gc.collect()


def load_model(model_path: str, n_channels: int, n_classes: int) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniMedXSeg(n_channels=n_channels, n_classes=n_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded on: {next(model.parameters()).device}")
    return model


def main():
    model_path = r"/root/autodl-tmp/best_dice_model.pth"
    n_channels, n_classes = 3, 1  # RGB input, single-class segmentation
    loaded_model = load_model(model_path, n_channels, n_classes)

    input_directory = r"/root/autodl-fs/pneumothorax-split/test/images"
    save_directory = r"/root/autodl-tmp/interpretable/pne"
    target_layers = ['down1', 'down2', 'down3', 'down4', 'up1', 'up2', 'up3', 'up4']

    # 批量处理，每5张图片清理一次内存
    batch_size = 5
    image_files = [f for f in os.listdir(input_directory) if f.endswith('.png')]

    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(image_files) + batch_size - 1) // batch_size}")

        for filename in batch:
            try:
                image_path = os.path.join(input_directory, filename)
                process_image(loaded_model, image_path, target_layers, save_directory)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

        # 每个批次后强制清理
        plt.close('all')
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()

