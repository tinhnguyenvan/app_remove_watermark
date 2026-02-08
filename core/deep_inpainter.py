"""
Deep learning-based inpainting module.
Uses a pre-trained model for high-quality watermark removal.
This is optional - falls back to OpenCV methods if not available.
"""

import cv2
import numpy as np
from loguru import logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Deep inpainting disabled.")


class InpaintingUNet(nn.Module):
    """Simple U-Net architecture for image inpainting."""

    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(4, 64)     # 4 channels: RGB + mask
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        # Output
        self.out_conv = nn.Conv2d(64, 3, 1)
        
        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self._pad_and_cat(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._pad_and_cat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._pad_and_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._pad_and_cat(d1, e1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.out_conv(d1))

    @staticmethod
    def _pad_and_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Handle size mismatches between upsampled and skip features."""
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        return torch.cat([x, skip], dim=1)


class DeepInpainter:
    """
    Deep learning-based inpainting for high-quality watermark removal.
    """

    MODEL_PATH = "models/inpainting_model.pth"

    def __init__(self, model_path: str | None = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for deep inpainting")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = InpaintingUNet().to(self.device)
        
        # Try to load pre-trained weights
        path = model_path or self.MODEL_PATH
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            logger.info(f"Loaded inpainting model from {path}")
        except FileNotFoundError:
            logger.warning(
                f"Model weights not found at {path}. "
                "Using randomly initialized model (results will be poor). "
                "Download pre-trained weights for better results."
            )

        self.model.eval()

    def inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint the masked region using the deep learning model.
        
        Args:
            frame: Input frame (BGR, uint8).
            mask: Binary mask (uint8, 255 = inpaint region).
            
        Returns:
            Inpainted frame.
        """
        h, w = frame.shape[:2]

        # Resize to model-friendly dimensions (multiple of 16)
        new_h = (h // 16) * 16
        new_w = (w // 16) * 16
        
        frame_resized = cv2.resize(frame, (new_w, new_h))
        mask_resized = cv2.resize(mask, (new_w, new_h))

        # Prepare input tensor: [B, 4, H, W] (RGB + mask)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
        mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0) / 255.0

        # Mask out the watermark region in input
        frame_tensor = frame_tensor * (1 - mask_tensor)
        
        input_tensor = torch.cat([frame_tensor, mask_tensor], dim=0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)

        # Convert output back to numpy
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_np = (output_np * 255).clip(0, 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

        # Resize back to original
        output_bgr = cv2.resize(output_bgr, (w, h))

        # Blend: only replace the masked region
        mask_3ch = cv2.cvtColor(
            cv2.resize(mask, (w, h)), cv2.COLOR_GRAY2BGR
        ).astype(np.float32) / 255.0
        
        result = (
            frame.astype(np.float32) * (1 - mask_3ch) + 
            output_bgr.astype(np.float32) * mask_3ch
        ).astype(np.uint8)

        return result
