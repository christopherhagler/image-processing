import logging
import os

from image import load_image, image_to_array, zero_order_hold, show_image, array_to_image, bilinear_interpolation
from PIL import Image

logger = logging.getLogger(__name__)


def part_0(image1: Image.Image, image2: Image.Image) -> None:
    image1_array = image_to_array(image1)
    image2_array = image_to_array(image2)

    row_index = 100
    column_index = 100

    extracted_block1 = image1_array[row_index:row_index + 128, column_index:column_index + 128]
    extracted_block2 = image2_array[row_index:row_index + 128, column_index:column_index + 128]

    show_image(
        array_to_image(
            zero_order_hold(
                extracted_block1,
                (512, 512)
            )
        )
    )

    show_image(
        array_to_image(
            bilinear_interpolation(
                extracted_block2,
                (512, 512)
            )
        )
    )


def part_1(image1: Image.Image, image2: Image.Image) -> None:
    pass


def part_2(image1: Image.Image, image2: Image.Image) -> None:
    pass


def part_3(image1: Image.Image, image2: Image.Image) -> None:
    pass


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

    part_0(image1, image2)
    # part_1(image1, image2)
    # part_2(image1, image2)
    # part_3(image1, image2)
