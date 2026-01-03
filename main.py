import argparse
import os

import cv2
import torch

from model import SRVGGNetCompact
from realesrgan_utils import RealESRGANer, unsharp_mask


def main():
    parser = argparse.ArgumentParser(description='Real-ESRGAN Image Enhancement')
    parser.add_argument('-i', '--input', type=str, default='input.jpg', help='Input image path')
    parser.add_argument('-o', '--output', type=str, default='output_4k.jpg', help='Output image path')
    parser.add_argument('-m', '--model_path', type=str, default='realesr-animevideov3.pth', help='Model path')
    parser.add_argument('-s', '--scale', type=int, default=4, help='Upscale factor (e.g., 2, 4)')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size to avoid OOM (e.g., 400). 0 for no tile')
    parser.add_argument('--denoise', type=float, default=0,
                        help='Denoising strength (h). Higher = smoother but less detail')
    parser.add_argument('--sharpen', type=float, default=1.5, help='Sharpening amount. Higher = sharper edges')
    parser.add_argument('--max_size', type=int, default=2560, help='Max output size (longest edge). 0 for no limit')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = SRVGGNetCompact(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=args.scale, act_type='prelu'
    )

    upsampler = RealESRGANer(
        scale=args.scale,
        model_path=args.model_path,
        model=model,
        tile=args.tile,
        tile_pad=10,
        pre_pad=0,
        half=device.type == 'cuda',
        device=device
    )

    print(f"Processing '{args.input}'...")
    img = cv2.imread(args.input)

    output, _ = upsampler.enhance(img, outscale=args.scale)

    if args.denoise > 0:
        print(f"Applying denoising (h={args.denoise})...")
        output = cv2.fastNlMeansDenoisingColored(
            output, None, h=args.denoise, hColor=args.denoise, templateWindowSize=7, searchWindowSize=21
        )

    if args.sharpen > 0:
        print(f"Applying sharpening (amount={args.sharpen})...")
        output = unsharp_mask(output, amount=args.sharpen)

    if args.max_size > 0:
        h, w = output.shape[:2]
        if max(h, w) > args.max_size:
            print(f"Resizing output to max {args.max_size}px...")
            scale_factor = args.max_size / max(h, w)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            output = cv2.resize(output, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cv2.imwrite(args.output, output)
    print(f'Done! Saved to {args.output}')


if __name__ == "__main__":
    main()
