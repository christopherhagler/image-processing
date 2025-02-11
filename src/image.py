import logging

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


def load_image(path: str) -> Image.Image:
    """
    Use Pillow to load an image specified by path

    :param path: the path point to the image to load
    :return: a pillow image object
    """
    try:
        if path:
            return Image.open(path)
    except Exception as e:
        logger.exception(e)
        raise e


def image_to_array(image: Image.Image) -> np.ndarray:
    """
    convert a pillow image object to a numpy ndarray

    :param image: the image to convert
    :return: a numpy ndarray representation of the image
    """
    try:
        if image:
            return np.array(image)
    except Exception as e:
        logger.exception(e)
        raise e


def fft2d_image(image: Image.Image) -> np.ndarray:
    """
    convert a pillow image object into a numpy array and perform the fast-fourier transform of the image.
    :param image: the image to convert into the Fourier domain
    :return: a numpy array containing the image frequencies
    """
    if image:
        image_array = image_to_array(image)
        return np.fft.fft2(image_array)

    raise ValueError("Cannot perform 2D-FFT on None.")


def remove_phase(fft_image: np.ndarray) -> np.ndarray:
    """
    Remove the phase of the image in the frequency domain
    :param fft_image: a numpy array containing the image frequencies
    :return: a numpy array containing on the frequency magnitudes
    """
    if fft_image is not None:
        return np.abs(fft_image)

    raise ValueError("Must supply an array of values to remove the phase of an image")


def remove_magnitude(fft_image: np.ndarray) -> np.ndarray:
    """
    Set the magnitudes of the image frequencies to 1.
    :param fft_image: a numpy array containing the image frequencies
    :return: a numpy array with all the frequency magnitudes set to 1.
    """
    if fft_image is not None:
        magnitude = np.abs(fft_image)
        normalized_magnitude = np.where(magnitude == 0, 1 + 0j, fft_image / magnitude)
        return normalized_magnitude

    raise ValueError("Must supply an array of values to remove the phase of an image")


def ifft2_image(fft_image: np.ndarray, shift: bool = False) -> np.ndarray:
    """
    perform the inverse fourier transform of an image to bring it back into the spatial domain.
    :param fft_image: the image in the frequency domain.
    :param shift: a boolean to indicate if an inverse shift should be performed before taking the inverse FFT.
    :return: a numpy array representing an image in the spatial domain.
    """
    shifted_image = fft_image
    if shift:
        shifted_image = np.fft.ifftshift(shifted_image, axes=None)

    return np.fft.ifft2(shifted_image)


def scale_image(image: np.ndarray, log_scale: bool = True) -> Image.Image:
    """
    Scale the axes of the image return it.
    :param image: the image to display
    :param log_scale: a boolean indicating if a log scale should be used
    :return: The normalized scaled version of the image.
    """
    spectrum = np.log(np.abs(image) + 1e-8)  # this fudge factor is to prevent log(0)
    if not log_scale:
        spectrum = np.abs(image)

    return array_to_image(spectrum)


def save_image(path: str, name: str, image: Image.Image) -> None:
    """
    Save an image to disk
    :param path: the path to save the image
    :param name: the name to give the image
    :param image: the image to save to disk
    """
    image.save(f"{path + name}")


def array_to_image(array: np.ndarray) -> Image.Image:
    """
    convert a numpy array representation of an image into an 8-bit quantized pillow image object.
    :param array: The array representation of the image.
    :return: a Pillow Image Object
    """
    mag_min = array.min()
    mag_max = array.max()

    magnitude_spectrum_normalized = (array - mag_min) / (mag_max - mag_min)
    magnitude_spectrum_normalized = (magnitude_spectrum_normalized * 255).astype(np.uint8)

    return Image.fromarray(magnitude_spectrum_normalized)


def show_image(image: Image.Image, title: str = None) -> None:
    """
    Display the image.
    :param image: The image to display
    :param title: the title to assign to the image
    """
    if image is not None:
        image.show(title=title)


def image_blur(fft_image: np.ndarray, blur: float = 0.8):
    """
    Low pass filter an image object in the frequency domain. This creates a circular mask where any frequencies
    greater than the cutoff frequency are set to zero.
    :param fft_image: this is a numpy representation of an image in the frequency domain.
    :param blur: a number of how much to blur the picture.
    :return: a numpy array representation of an image that has been blurred (Low Pass Filtered)
    """
    m, n = fft_image.shape
    omega_max = 0.5 * min(m, n)
    cutoff_frequency = omega_max * blur

    # build a mask to hold the low pass filter values
    low_pass = np.zeros((m, n), dtype=np.float32)

    # Coordinates of the center
    x = m // 2
    y = n // 2

    for i in range(m):
        for j in range(n):
            dist = np.sqrt((i - x) ** 2 + (j - y) ** 2)
            if dist <= cutoff_frequency:
                low_pass[i, j] = 1.0

    filtered_image = np.fft.fftshift(fft_image) * low_pass
    return np.fft.ifftshift(filtered_image, axes=None)
