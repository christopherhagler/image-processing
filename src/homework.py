import logging
import os

from image import *
from PIL import Image

logger = logging.getLogger(__name__)


def create_noisy_images(image1: Image.Image) -> None:
    guassian = add_gaussian_noise(image_to_array(image=image1))
    salt_and_pepper = add_salt_pepper_noise(image_to_array(image=image1))

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="lena_g.png",
        image=array_to_image(guassian)
    )

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="lena_s.png",
        image=array_to_image(salt_and_pepper)
    )


def run_homework_steps():
    image1_path = str(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'resources/images',
        'lena.raw'
    ))

    image2_path = str(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'resources/images',
        'lena.png'
    ))

    lena_raw = load_raw_image(image1_path)
    lena_png = load_image(image2_path)

    create_noisy_images(lena_raw)
