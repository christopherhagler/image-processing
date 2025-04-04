import logging
import os

import matplotlib.pyplot as plt
from image import *
from PIL import Image

logger = logging.getLogger(__name__)


def part_1(image1: Image.Image) -> None:
    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="lena_gamma.png",
        image=array_to_image(inverse_gamma_correction(image_to_array(image1)))
    )


def part_2_and_3(image1: Image.Image) -> None:
    bin_values = [2, 128, 256]

    gamma_path = str(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'resources/images/generated',
        'lena_gamma.png'
    ))
    gamma_image = load_image(gamma_path)

    for bin_value in bin_values:
        histogram = calculate_histogram(image_to_array(image1), bins=bin_value)
        bins = np.arange(start=0, stop=255, step=255 / bin_value)

        plt.bar(bins, histogram, width=1.0, edgecolor='black')
        plt.xlabel('Bin Index')
        plt.ylabel('Count')
        plt.title('Lena.png Histogram Plot')
        plt.show()

        gamma_histogram = calculate_histogram(image_to_array(gamma_image), bins=bin_value)
        plt.bar(bins, gamma_histogram, width=1.0, edgecolor='black')
        plt.xlabel('Bin Index')
        plt.ylabel('Count')
        plt.title('Gamma Corrected Histogram Plot')
        plt.show()

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="lena_histogram_normalized.png",
        image=array_to_image(normalized_histogram(image_to_array(gamma_image), bins=256))
    )


def part_4_and_5(image1: Image.Image) -> None:
    gaussian = gaussian_noise(image_to_array(image=image1))
    salt_and_pepper = salt_pepper_noise(image_to_array(image=image1))

    # Noisy Images
    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="lena_g.png",
        image=array_to_image(gaussian)
    )

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="lena_s.png",
        image=array_to_image(salt_and_pepper)
    )

    for lobe_sz in [0.25, 0.5, 1.0]:
        # Linear Filter
        save_image(
            path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
            name="lena_g_smoothed.png",
            image=array_to_image(gaussian_blur(gaussian, sigma=lobe_sz))
        )

        save_image(
            path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
            name="lena_s_smoothed.png",
            image=array_to_image(gaussian_blur(salt_and_pepper, sigma=lobe_sz))
        )

    # Median Filter
    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="lena_s_median_1.png",
        image=array_to_image(median_filter(image=salt_and_pepper, n_times=1))
    )

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="lena_g_median_1.png",
        image=array_to_image(median_filter(image=gaussian))
    )

    save_image(
        path=os.path.dirname(os.path.abspath(__file__)) + "/resources/images/generated/",
        name="lena_s_median_50.png",
        image=array_to_image(median_filter(image=salt_and_pepper, n_times=50))
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

    part_1(lena_png)
    part_2_and_3(lena_png)
    part_4_and_5(lena_raw)
