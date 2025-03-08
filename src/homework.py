import logging
import os

from image import load_image, image_to_array, zero_order_hold, show_image, array_to_image, bilinear_interpolation, \
    save_image, decimation
from PIL import Image

logger = logging.getLogger(__name__)


def part_1a(image1: Image.Image, image2: Image.Image) -> None:
    image2_array = image_to_array(image2)

    start_position2 = (120, 120)

    extracted_block2 = image2_array[
        start_position2[0]:start_position2[0] + 128,
        start_position2[1]:start_position2[1] + 128
    ]

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image2_zero_order_hold.png",
        image=array_to_image(
            zero_order_hold(
                extracted_block2,
                (512, 512)
            )
        )
    )


def part_1b(image1: Image.Image, image2: Image.Image) -> None:
    image2_array = image_to_array(image2)

    start_position2 = (120, 120)

    extracted_block2 = image2_array[
        start_position2[0]:start_position2[0] + 128,
        start_position2[1]:start_position2[1] + 128
    ]

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image2_bilinear_interpolation.png",
        image=array_to_image(
            bilinear_interpolation(
                extracted_block2,
                (512, 512)
            )
        )
    )


def part_2a(image1: Image.Image, image2: Image.Image) -> None:
    image2_array = image_to_array(image2)

    decimated_image = decimation(image2_array)
    restored_image = bilinear_interpolation(decimated_image, (512, 512))
    # difference_image =

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image2_decimation_M2.png",
        image=array_to_image(decimated_image)
    )
    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image2_bilinear_restored_M2.png",
        image=array_to_image(restored_image)
    )


def part_2b(image1: Image.Image, image2: Image.Image) -> None:
    image2_array = image_to_array(image2)

    decimated_image = decimation(image2_array)
    restored_image = bilinear_interpolation(decimated_image, (512, 512))
    # difference_image =

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image2_low_pass_decimation_M2.png",
        image=array_to_image(decimated_image)
    )
    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image2_low_pass_bilinear_restored_M2.png",
        image=array_to_image(restored_image)
    )


def part_2c(image1: Image.Image, image2: Image.Image) -> None:
    image2_array = image_to_array(image2)

    decimated_image = decimation(image2_array)
    restored_image = bilinear_interpolation(decimated_image, (512, 512))
    # difference_image =

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image2_decimation_M8.png",
        image=array_to_image(decimated_image)
    )
    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image2_bilinear_restored_M8.png",
        image=array_to_image(restored_image)
    )


def part_2d(image1: Image.Image, image2: Image.Image) -> None:
    image2_array = image_to_array(image2)

    decimated_image = decimation(image2_array, factor=8)
    restored_image = bilinear_interpolation(decimated_image, (512, 512))
    # difference_image =

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image2_low_pass_decimation_M8.png",
        image=array_to_image(decimated_image)
    )
    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image2_low_pass_bilinear_restored_M8.png",
        image=array_to_image(restored_image)
    )


def run_homework_steps():
    image1_path = str(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'resources/images',
        'cat_thoughts.png'
    ))

    image2_path = str(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'resources/images',
        'top_student.png'
    ))

    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    part_1a(image1, image2)
    part_1b(image1, image2)
    part_2a(image1, image2)
    part_2b(image1, image2)
    part_2c(image1, image2)
    part_2d(image1, image2)
