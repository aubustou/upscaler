from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale import (
    StableDiffusionUpscalePipeline,
)
from PIL import Image
from PIL.Image import Resampling
from torch.cuda import OutOfMemoryError

logger = logging.getLogger(__name__)

model_id = "stabilityai/stable-diffusion-x4-upscaler"
UPSCALED_SUFFIX = "_upscaled"
TEMPORARY_SUFFIX = "_temp"

MAX_IMAGE_WIDTH = 500
MAX_IMAGE_HEIGHT = 500
MAX_NUMBER_OF_PIXELS = MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT


def cut_image(image: Path) -> list[Path]:
    """If the image is bigger than a given size, cut it into multiple, overlapping parts"""
    width = MAX_IMAGE_WIDTH
    height = MAX_IMAGE_HEIGHT

    image_data = Image.open(image)

    parts: list[Path] = []
    for i in range(0, image_data.width, width // 2):
        for j in range(0, image_data.height, height // 2):
            part = image_data.crop(
                (
                    i,
                    j,
                    min(i + width, image_data.width),
                    min(j + height, image_data.height),
                )
            )
            path = image.parent / (image.stem + f"_{i}_{j}" + TEMPORARY_SUFFIX + ".jpg")
            part.save(path)
            parts.append(path)

    logger.info("Cut image into %d parts", len(parts))

    return parts


def recompose_image_from_paths(
    parts: list[Path], width: int, height: int
) -> Image.Image:
    """Recompose the image from the parts"""

    return recompose_image([Image.open(part) for part in parts], width, height)


def recompose_image(parts: list[Image.Image], width: int, height: int) -> Image.Image:
    """Recompose the image from the parts"""

    # Upscale the width and height
    width = width * 4
    height = height * 4
    max_width = MAX_IMAGE_WIDTH * 2
    max_height = MAX_IMAGE_HEIGHT * 2

    logger.info("Recomposing image from %d parts", len(parts))

    image = Image.new("RGB", (width, height), (0, 0, 0))
    for i in range(0, width, max_width):
        for j in range(0, height, max_height):
            part = parts.pop(0)
            image.paste(part, (i, j))

    return image


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
                generator = input_folder.rglob(f"*.{extension}")
            else:
                generator = input_folder.glob(f"*.{extension}")
            breakpoint()
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
    logger.info("Upscale: %s", upscale)

    logger.info("Input folder: %s", input_folder)

    if chosen_output_folder := args.output_folder:
        chosen_output_folder = chosen_output_folder / "upscaled"
        logger.info("Output folder: %s", chosen_output_folder)
    else:
        logger.info("Output folder is not provided")

    overwrite = args.overwrite

    gpu_number = args.gpu_number
    logger.info("Set to GPU %d", gpu_number)
    device = f"cuda:{gpu_number}"

    if enable_attention_slicing := args.enable_attention_slicing:
        logger.info("Attention slicing enabled")
    else:
        logger.info("Attention slicing disabled")

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
    enable_xformers_memory_attention: bool = True,
    device: str = "cuda",
    seed_generator: torch.Generator | None = None,
):
    image_data = Image.open(image)
    if image_data.mode != "RGB":
        logger.info("Converting image to RGB")
        image_data = image_data.convert("RGB")

    logger.info("Upscaling %s", image)

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

    logger.info("Image size: %dx%d", image_data.width, image_data.height)
    logger.info("Target size: %dx%d", resize_x, resize_y)

    number_of_pixels = image_data.width * image_data.height

    if upscale:
        if number_of_pixels > MAX_NUMBER_OF_PIXELS:
            logger.info("Image is too big. Cutting it into parts")

            parts: list[Path] = cut_image(image)
            seed = random.randint(0, 2**32 - 1)
            seed_generator = torch.Generator(device=device).manual_seed(seed)

            logger.info("Use seed %d", seed)

            for image_ in parts:
                treat_image(
                    image_,
                    chosen_output_folder,
                    prompt,
                    overwrite=overwrite,
                    upscale=upscale,
                    resize=resize,
                    resize_x=resize_x,
                    resize_y=resize_y,
                    full_frame_resize_x=full_frame_resize_x,
                    full_frame_resize_y=full_frame_resize_y,
                    enable_attention_slicing=enable_attention_slicing,
                    enable_xformers_memory_attention=enable_xformers_memory_attention,
                    device=device,
                    seed_generator=seed_generator,
                )
            image_data = recompose_image(parts, image_data.width, image_data.height)
        # Upscale only if the image is smaller than the target size
        elif image_data.width < resize_x or image_data.height < resize_y:
            logger.info("Loading model %s", model_id)
            torch.cuda.empty_cache()
            pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                model_id, torch_dtype=torch.float16
            )
            pipeline = pipeline.to(device)
            logger.info("Processing %s", image)
            if enable_attention_slicing:
                pipeline.enable_attention_slicing()
            elif enable_xformers_memory_attention:
                pipeline.enable_xformers_memory_efficient_attention()

            try:
                image_data = pipeline(
                    prompt=prompt,
                    image=image_data,
                    generator=[seed_generator] if seed_generator is not None else None,
                ).images[0]
            except OutOfMemoryError as exc:
                logger.warning(exc)
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
                        seed_generator=seed_generator,
                    )
                else:
                    logging.error("Failed to upscale %s. Passing", image)
                    return

    if full_frame_resize_x is not None and full_frame_resize_y is not None:
        image_data = resize_image(
            image_data,
            (resize_x, resize_y),
            (full_frame_resize_x, full_frame_resize_y),
        )
    elif resize:
        image_data = image_data.resize((resize_x, resize_y), Resampling.LANCZOS)
    logger.info("Saving %s", hi_res_img_path)
    image_data.save(hi_res_img_path)


def resize_image(
    image: Image.Image,
    resize: tuple[int, int],
    full_frame_resize: tuple[int, int],
) -> Image.Image:
    if resize:
        logger.info("Resizing %s", resize)
        image = image.resize(resize, Resampling.LANCZOS)
    logger.info("Resizing to full frame %s", full_frame_resize)
    background = Image.new("RGB", full_frame_resize, (0, 0, 0))
    x = (background.width - image.width) // 2
    y = (background.height - image.height) // 2
    background.paste(image, (x, y))

    return background


if __name__ == "__main__":
    main()
    # parts = cut_image(Path("95df8f14c935c44fdd6f353966974f81.jpg"))
    # parts = [
    #     Path("upscaled/de-funes_0_0_temp_upscaled.jpg"),
    #     Path("upscaled/de-funes_0_250_temp_upscaled.jpg"),
    #     Path("upscaled/de-funes_0_500_temp_upscaled.jpg"),
    #     Path("upscaled/de-funes_250_0_temp_upscaled.jpg"),
    #     Path("upscaled/de-funes_250_250_temp_upscaled.jpg"),
    #     Path("upscaled/de-funes_250_500_temp_upscaled.jpg"),
    #     Path("upscaled/de-funes_500_0_temp_upscaled.jpg"),
    #     Path("upscaled/de-funes_500_250_temp_upscaled.jpg"),
    #     Path("upscaled/de-funes_500_500_temp_upscaled.jpg"),
    #     Path("upscaled/de-funes_750_0_temp_upscaled.jpg"),
    #     Path("upscaled/de-funes_750_250_temp_upscaled.jpg"),
    #     Path("upscaled/de-funes_750_500_temp_upscaled.jpg"),
    # ]
    # image = recompose_image_from_paths(parts, 960, 720)
    # image.show()
    # image.save("recomposed.jpg")
