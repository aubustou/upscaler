from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale import (
    StableDiffusionUpscalePipeline,
)
from PIL import Image
from PIL.Image import Resampling
from torch.cuda import OutOfMemoryError

model_id = "stabilityai/stable-diffusion-x4-upscaler"
UPSCALED_SUFFIX = "_upscaled"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--image-folder", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--output-folder", type=Path, default=None)
    parser.add_argument("--resize", type=str, default=None)
    parser.add_argument("--no-upscale", action="store_true")
    parser.add_argument("--full-frame-resize", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--gpu-number", type=int, default=0)
    parser.add_argument("--enable-attention-slicing", "-e", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not (args.image_folder or args.image):
        raise ValueError("Either image or image folder must be provided")

    prompt = args.prompt

    images: list[Path] = []
    if args.image_folder is not None:
        input_folder = args.image_folder
        for extension in ["jpg", "png"]:
            if args.recursive:
                generator = args.image_folder.rglob(f"*.{extension}")
            else:
                generator = args.image_folder.glob(f"*.{extension}")
            images.extend([x for x in generator if UPSCALED_SUFFIX not in x.name])
        images.sort()
    else:
        input_folder = args.image.parent
        images = [args.image]

    if resize := args.resize:
        resize_x, resize_y = (int(x) for x in resize.split("x"))
    else:
        resize_x = resize_y = None

    if full_frame_resize := args.full_frame_resize:
        full_frame_resize_x, full_frame_resize_y = (
            int(x) for x in full_frame_resize.split("x")
        )
    else:
        full_frame_resize_x = full_frame_resize_y = None

    upscale = not args.no_upscale

    logging.info("Input folder: %s", input_folder)

    if chosen_output_folder := args.output_folder:
        chosen_output_folder = chosen_output_folder / "upscaled"
        logging.info("Output folder: %s", chosen_output_folder)
    else:
        logging.info("Output folder is not provided")

    overwrite = args.overwrite

    gpu_number = args.gpu_number
    logging.info("Set to GPU %d", gpu_number)
    device = f"cuda:{gpu_number}"

    enable_attention_slicing = args.enable_attention_slicing
    logging.info("Enable attention slicing: %s", enable_attention_slicing)

    for image in images:
        treat_image(
            image,
            chosen_output_folder,
            prompt,
            overwrite=overwrite,
            upscale=upscale,
            resize=bool(resize),
            resize_x=resize_x,
            resize_y=resize_y,
            full_frame_resize_x=full_frame_resize_x,
            full_frame_resize_y=full_frame_resize_y,
            device=device,
            enable_attention_slicing=enable_attention_slicing,
        )


def treat_image(
    image: Path,
    chosen_output_folder: Path,
    prompt: str,
    *,
    overwrite: bool,
    upscale: bool,
    resize: bool,
    resize_x: int | None,
    resize_y: int | None,
    full_frame_resize_x: int | None,
    full_frame_resize_y: int | None,
    enable_attention_slicing: bool = False,
    device: str = "cuda",
):
    image_data = Image.open(image).convert("RGB")

    if not chosen_output_folder:
        output_folder = image.parent / "upscaled"
    else:
        output_folder = chosen_output_folder

    output_folder.mkdir(exist_ok=True, parents=True)
    hi_res_img_path = output_folder / (image.stem + UPSCALED_SUFFIX + ".jpg")
    if hi_res_img_path.exists() and not overwrite:
        return

    if resize_x is None:
        resize_x = image_data.width * 4
    if resize_y is None:
        resize_y = image_data.height * 4

    # Upscale only if the image is smaller than the target size
    if upscale and (image_data.width < resize_x or image_data.height < resize_y):
        logging.info("Loading model")
        torch.cuda.empty_cache()
        pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        pipeline = pipeline.to(device)
        logging.info("Processing %s", image)
        if enable_attention_slicing:
            pipeline.enable_attention_slicing()

        try:
            image_data = pipeline(prompt=prompt, image=image_data).images[0]
        except OutOfMemoryError as exc:
            logging.warning(exc)
            if not enable_attention_slicing:
                return treat_image(
                    image,
                    chosen_output_folder,
                    prompt,
                    overwrite=overwrite,
                    upscale=upscale,
                    resize=resize,
                    resize_x=resize_x,
                    resize_y=resize_y,
                    full_frame_resize_x=full_frame_resize_x,
                    full_frame_resize_y=full_frame_resize_y,
                    enable_attention_slicing=True,
                )
            else:
                raise

    if full_frame_resize_x is not None and full_frame_resize_y is not None:
        image_data = resize_image(
            image_data,
            (resize_x, resize_y),
            (full_frame_resize_x, full_frame_resize_y),
        )
    elif resize:
        image_data = image_data.resize((resize_x, resize_y), Resampling.LANCZOS)
    logging.info("Saving %s", hi_res_img_path)
    image_data.save(hi_res_img_path)


def resize_image(
    image: Image.Image,
    resize: tuple[int, int],
    full_frame_resize: tuple[int, int],
) -> Image.Image:
    if resize:
        logging.info("Resizing %s", resize)
        image = image.resize(resize, Resampling.LANCZOS)
    logging.info("Resizing to full frame %s", full_frame_resize)
    background = Image.new("RGB", full_frame_resize, (0, 0, 0))
    x = (background.width - image.width) // 2
    y = (background.height - image.height) // 2
    background.paste(image, (x, y))

    return background


if __name__ == "__main__":
    main()
