import logging
import os

from image import load_image, fft2d_image, ifft2_image, array_to_image, show_image, remove_phase, \
    remove_magnitude, image_blur, scale_image
from PIL import Image

logger = logging.getLogger(__name__)


def part_0(roll_image: Image.Image, univ_image: Image.Image) -> None:
    logger.info(
        "Running part 0 of the homework. The following operations performed in this step are: \n"
        "1). Read in the images \n"
        "2). Take the images into the frequency domain using the FFT.\n"
        "3). Take the images back to the spatial domain using the IFFT. \n"
        "4). Display the magnitude of the Roll.png in the frequency domain on normal scale. \n"
        "5). Display the magnitude of the Roll.png in the frequency domain on log scale. \n"
        "6). Display the magnitude of the Univ.png in the frequency domain on normal scale. \n"
        "7). Display the magnitude of the Univ.png in the frequency domain on log scale. \n"
    )

    # take the image into the frequency domain
    roll_fft = fft2d_image(roll_image)
    univ_fft = fft2d_image(univ_image)

    # bring the image back to the spatial domain
    inverse_roll_image = ifft2_image(roll_fft)
    inverse_univ_image = ifft2_image(univ_fft)

    # Show the frequency domain images
    show_image(scale_image(roll_fft, log_scale=False), title="Roll Image Magnitude Spectrum")
    show_image(scale_image(roll_fft, log_scale=True), title="Roll Image Log Spectrum")

    show_image(scale_image(univ_fft, log_scale=False), title="Univ Image Magnitude Spectrum")
    show_image(scale_image(univ_fft, log_scale=True), title="Univ Image Log Spectrum")


def part_1(roll_image: Image.Image, univ_image: Image.Image) -> None:
    logger.info(
        "Running part 1 of the homework. The following operations performed in this step are: \n"
        "1). Read in the images \n"
        "2). Take the images into the frequency domain using the FFT.\n"
        "3). While the images are in the frequency domain, remove the phase. \n"
        "4). Take the images back to the spatial domain using the IFFT. \n"
        "5). Display the images in the spatial domain. \n"
    )

    # take the image into the frequency domain
    roll_fft = fft2d_image(roll_image)
    univ_fft = fft2d_image(univ_image)

    # bring the image back to the spatial domain with the phase removed
    inverse_roll_image = ifft2_image(remove_phase(roll_fft))
    inverse_univ_image = ifft2_image(remove_phase(univ_fft))

    # Display the images
    show_image(scale_image(inverse_roll_image, log_scale=True), title="Roll Image")
    show_image(scale_image(inverse_univ_image, log_scale=True), title="Univ Image")


def part_2(roll_image: Image.Image, univ_image: Image.Image) -> None:
    logger.info(
        "Running part 2 of the homework. The following operations performed in this step are: \n"
        "1). Read in the images \n"
        "2). Take the images into the frequency domain using the FFT.\n"
        "3). While the images are in the frequency domain, set the magnitude to 1 while preserving the phase. \n"
        "4). Take the images back to the spatial domain using the IFFT. \n"
        "5). Display the images in the spatial domain. \n"
    )

    # take the image into the frequency domain
    roll_fft = fft2d_image(roll_image)
    univ_fft = fft2d_image(univ_image)

    # bring the image back to the spatial domain with the phase removed
    inverse_roll_image = ifft2_image(remove_magnitude(roll_fft))
    inverse_univ_image = ifft2_image(remove_magnitude(univ_fft))

    # Display the images
    show_image(scale_image(inverse_roll_image, log_scale=False), title="Roll Image")
    show_image(scale_image(inverse_univ_image, log_scale=False), title="Univ Image")


def part_3(roll_image: Image.Image, univ_image: Image.Image) -> None:
    logger.info(
        "Running part 3 of the homework. The following operations performed in this step are: \n"
        "1). Read in the images \n"
        "2). Take the images into the frequency domain using the FFT.\n"
        "3). While the images are in the frequency domain, low pass filter the images. \n"
        "4). Take the images back to the spatial domain using the IFFT. \n"
        "5). Display the images in the spatial domain. \n"
    )

    # take the image into the frequency domain
    roll_fft = fft2d_image(roll_image)
    univ_fft = fft2d_image(univ_image)

    # filter the image with |omega| > 0.5*omega_max = 0
    roll_filtered = image_blur(roll_fft, blur=0.8)
    univ_filtered = image_blur(univ_fft, blur=0.8)

    # bring the image back to the spatial domain with the phase removed
    inverse_roll_image = ifft2_image(roll_filtered)
    inverse_univ_image = ifft2_image(univ_filtered)

    # Display the images
    show_image(scale_image(inverse_roll_image, log_scale=False), title="Roll Image")
    show_image(scale_image(inverse_univ_image, log_scale=False), title="Univ Image")


def run_homework_steps():
    roll_image_path = str(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'resources',
        'roll.png'
    ))

    univ_image_path = str(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'resources',
        'univ.png'
    ))

    roll_image = load_image(roll_image_path)
    univ_image = load_image(univ_image_path)

    part_0(roll_image, univ_image)
    part_1(roll_image, univ_image)
    part_2(roll_image, univ_image)
    part_3(roll_image, univ_image)
