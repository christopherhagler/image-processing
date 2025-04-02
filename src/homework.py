import logging
import os

from image import *
from PIL import Image

logger = logging.getLogger(__name__)


def part_1a(image1: Image.Image, image2: Image.Image) -> None:
    image1_array = image_to_array(image1)
    image2_array = image_to_array(image2)

    start_position1 = (0, 0)
    start_position2 = (120, 120)

    extracted_block1 = image1_array[
                       start_position1[0]:start_position1[0] + 128,
                       start_position1[1]:start_position1[1] + 128
    ]
    extracted_block2 = image2_array[
                       start_position2[0]:start_position2[0] + 128,
                       start_position2[1]:start_position2[1] + 128
    ]

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image1_zero_order_hold.png",
        image=array_to_image(
            zero_order_hold(
                extracted_block1,
                (512, 512)
            )
        )
    )
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
    image1_array = image_to_array(image1)
    image2_array = image_to_array(image2)

    start_position1 = (0, 0)
    start_position2 = (120, 120)

    extracted_block1 = image1_array[
                       start_position1[0]:start_position1[0] + 128,
                       start_position1[1]:start_position1[1] + 128
    ]
    extracted_block2 = image2_array[
                       start_position2[0]:start_position2[0] + 128,
                       start_position2[1]:start_position2[1] + 128
    ]

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image1_bilinear_interpolation.png",
        image=array_to_image(
            bilinear_interpolation(
                extracted_block1,
                (512, 512)
            )
        )
    )

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
    image1_array = image_to_array(image1)
    image2_array = image_to_array(image2)

    decimated_image1 = decimation(image1_array)
    restored_image1 = bilinear_interpolation(decimated_image1, (512, 512))
    diff_image1 = difference_image(image1_array, restored_image1)
    print(f"Mean Squared Error part 2A cat thoughts: {mean_squared_error(image1_array, restored_image1)}")

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image1_decimation_M2_cat_thoughts.png",
        image=array_to_image(decimated_image1)
    )
    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image1_bilinear_restored_M2_cat_thoughts.png",
        image=array_to_image(restored_image1)
    )

    decimated_image2 = decimation(image2_array)
    restored_image2 = bilinear_interpolation(decimated_image2, (512, 512))
    diff_image2 = difference_image(image2_array, restored_image2)
    print(f"Mean Squared Error part 2A top student: {mean_squared_error(image2_array, restored_image2)}")

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image2_decimation_M2_top_student.png",
        image=array_to_image(decimated_image2)
    )
    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image2_bilinear_restored_M2_top_student.png",
        image=array_to_image(restored_image2)
    )


def part_2b(image1: Image.Image, image2: Image.Image) -> None:
    image1_array = image_to_array(image1)
    image2_array = image_to_array(image2)

    decimated_image1 = low_pass_decimation(image1_array)
    restored_image1 = bilinear_interpolation(decimated_image1, (512, 512))
    diff_image1 = difference_image(image1_array, restored_image1)

    print(f"Mean Squared Error part 2B cat thoughts: {mean_squared_error(image1_array, restored_image1)}")

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image1_low_pass_decimation_M2_cat_thoughts.png",
        image=array_to_image(decimated_image1)
    )
    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image1_low_pass_bilinear_restored_M2_cat_thoughts.png",
        image=array_to_image(restored_image1)
    )

    decimated_image2 = low_pass_decimation(image2_array)
    restored_image2 = bilinear_interpolation(decimated_image2, (512, 512))
    diff_image2 = difference_image(image2_array, restored_image2)

    print(f"Mean Squared Error part 2B top student: {mean_squared_error(image2_array, restored_image2)}")

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image2_low_pass_decimation_M2_top_student.png",
        image=array_to_image(decimated_image2)
    )
    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image2_low_pass_bilinear_restored_M2_top_student.png",
        image=array_to_image(restored_image2)
    )


def part_2c(image1: Image.Image, image2: Image.Image) -> None:
    image1_array = image_to_array(image1)
    image2_array = image_to_array(image2)

    decimated_image1 = decimation(image1_array, factor=8)
    restored_image1 = bilinear_interpolation(decimated_image1, (512, 512))
    diff_image1 = difference_image(image1_array, restored_image1)
    print(f"Mean Squared Error part 2C cat thoughts: {mean_squared_error(image1_array, restored_image1)}")

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image1_decimation_M8_cat_thoughts.png",
        image=array_to_image(decimated_image1)
    )
    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image1_bilinear_restored_M8_cat_thoughts.png",
        image=array_to_image(restored_image1)
    )

    decimated_image2 = decimation(image2_array, factor=8)
    restored_image2 = bilinear_interpolation(decimated_image2, (512, 512))
    diff_image2 = difference_image(image2_array, restored_image2)
    print(f"Mean Squared Error part 2C top student: {mean_squared_error(image2_array, restored_image2)}")

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image2_decimation_M8_top_student.png",
        image=array_to_image(decimated_image2)
    )
    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image2_bilinear_restored_M8_top_student.png",
        image=array_to_image(restored_image2)
    )


def part_2d(image1: Image.Image, image2: Image.Image) -> None:
    image1_array = image_to_array(image1)
    image2_array = image_to_array(image2)

    decimated_image1 = low_pass_decimation(image1_array, factor=8)
    restored_image1 = bilinear_interpolation(decimated_image1, (512, 512))
    diff_image1 = difference_image(image1_array, restored_image1)
    print(f"Mean Squared Error part 2D cat thoughts: {mean_squared_error(image1_array, restored_image1)}")

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image1_low_pass_decimation_M8.png",
        image=array_to_image(decimated_image1)
    )
    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image1_low_pass_bilinear_restored_M8.png",
        image=array_to_image(restored_image1)
    )

    decimated_image2 = low_pass_decimation(image2_array, factor=8)
    restored_image2 = bilinear_interpolation(decimated_image2, (512, 512))
    diff_image2 = difference_image(image2_array, restored_image2)
    print(f"Mean Squared Error part 2D top student: {mean_squared_error(image2_array, restored_image2)}")

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image2_low_pass_decimation_M8.png",
        image=array_to_image(decimated_image2)
    )
    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="image2_low_pass_bilinear_restored_M8.png",
        image=array_to_image(restored_image2)
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
