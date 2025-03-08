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


def image_blur(fft_image: np.ndarray, blur: float = 0.5):
    """
    Low pass filter an image object in the frequency domain. This creates a circular mask where any frequencies
    greater than the cutoff frequency are set to zero.
    :param fft_image: this is a numpy representation of an image in the frequency domain.
    :param blur: a number of how much to blur the picture.
    :return: a numpy array representation of an image that has been blurred (Low Pass Filtered)
    """
    m, n = fft_image.shape
    omega_max = blur * np.sqrt((m / 2) ** 2 + (n / 2) ** 2)

    # build a mask to hold the low pass filter values
    low_pass = np.zeros((m, n), dtype=np.float32)

    # Coordinates of the center
    x = m // 2
    y = n // 2

    for i in range(m):
        for j in range(n):
            dist = np.sqrt((i - x) ** 2 + (j - y) ** 2)
            if dist <= omega_max:
                low_pass[i, j] = 1.0

    filtered_image = np.fft.fftshift(fft_image) * low_pass
    return np.fft.ifftshift(filtered_image, axes=None)


def zero_order_hold(
        array: np.ndarray,
        dimensions: tuple[int, int] = None
) -> np.ndarray:
    """
    Magnify an image using the zero-order hold technique.
    :param array: The original image(or section) to magnify.
    :param dimensions: The dimensions of the new image.
    :return: Numpy NDArray that contains the magnified image.
    """
    if not dimensions:
        raise ValueError("In order to magnify an image, the magnification dimensions must be specified.")

    m, n = array.shape
    if m == 0 or n == 0:
        raise ValueError("Cannot magnify an image that is of zero length.")

    if m > dimensions[0] or n > dimensions[1]:
        raise ValueError("The dimensions of the magnified image cannot be smaller than the original image.")

    m_repeat = dimensions[0] // m
    n_repeat = dimensions[1] // n
    return np.repeat(
        np.repeat(array, m_repeat, axis=0),
        n_repeat,
        axis=1
    )


def bilinear_interpolation(
        array: np.ndarray,
        dimensions: tuple[int, int] = None
) -> np.ndarray:
    if not dimensions:
        raise ValueError("In order to magnify an image, the magnification dimensions must be specified.")

    m, n = array.shape
    if m == 0 or n == 0:
        raise ValueError("Cannot magnify an image that is of zero length.")

    if m > dimensions[0] or n > dimensions[1]:
        raise ValueError("The dimensions of the magnified image cannot be smaller than the original image.")

    row_spacing = dimensions[0] // m
    column_spacing = dimensions[1] // n

    # create the zero padded image
    magnified_image = np.zeros(dimensions)
    magnified_image[::row_spacing, ::column_spacing] = array

    # Interpolate within the rows
    for i in range(0, dimensions[0], row_spacing):
        for j in range(0, dimensions[1], column_spacing):
            a = magnified_image[i, j]
            if j + column_spacing > dimensions[1] - 1:
                b = a
            else:
                b = magnified_image[i, j + column_spacing]

            for k in range(j + 1, j + column_spacing):
                # No need to try and interpolate outside the boundary of the magnified image
                if k > dimensions[1]:
                    break

                magnified_image[i, k] = ((1 - ((k - j) / column_spacing)) * a) + (((k - j) / column_spacing) * b)

    # Interpolate within the columns
    for i in range(0, dimensions[0], row_spacing):
        for j in range(0, dimensions[1]):
            a = magnified_image[i, j]
            if i + row_spacing > dimensions[0] - 1:
                b = a
            else:
                b = magnified_image[i + row_spacing, j]

            for k in range(i + 1, i + row_spacing):
                # No need to try and interpolate outside the boundary of the magnified image
                if k > dimensions[0]:
                    break

                magnified_image[k, j] = ((1 - ((k - i) / row_spacing)) * a) + (((k - i) / row_spacing) * b)

    return magnified_image


def decimation(
        array: np.ndarray,
        factor: int = 2
) -> np.ndarray:
    """
    Decimate an image by whatever factor is passed in. This method does not low pass filter the image first, so
    there is a good chance of aliasing.
    :param array:
    :param factor:
    :return:
    """
    m, n = array.shape

    # Make the process easy for now
    if m != n:
        raise ValueError("Only NxN images are supported for decimation.")

    img_sz = (m // factor, n // factor)
    small_image = np.zeros(shape=img_sz)

    small_image_row_index = 0
    for i in range(0, m, factor):

        small_img_column_index = 0
        for j in range(0, n, factor):
            pixel = array[i, j]

            small_image[small_image_row_index, small_img_column_index] = pixel
            small_img_column_index += 1

        small_image_row_index += 1

    return small_image


def low_pass_decimate(
        array: np.ndarray,
        factor: int = 2
) -> np.ndarray:
    """
        Decimate an image by whatever factor is passed in and low-pass filter the image first to prevent aliasing.
        :param array:
        :param factor:
        :return:
        """
    m, n = array.shape

    # Make the process easy for now
    if m != n:
        raise ValueError("Only NxN images are supported for decimation.")

    img_sz = (m // factor, n // factor)
    small_image = np.zeros(shape=img_sz)

    small_image_row_index = 0
    for i in range(0, m, factor):

        small_img_column_index = 0
        for j in range(0, n, factor):
            pixel = array[i, j]

            small_image[small_image_row_index, small_img_column_index] = pixel
            small_img_column_index += 1

        small_image_row_index += 1

    return small_image
