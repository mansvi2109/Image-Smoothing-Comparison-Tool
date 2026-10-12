import cv2
import numpy as np
import gradio as gr


# Add Salt-and-Pepper Noise
def add_salt_pepper_noise(image, amount=0.05):
    noisy = image.copy()

    height, width = image.shape[:2]
    total_pixels = height * width
    number_of_pixels = int(amount * total_pixels)

    # Salt noise
    salt_y = np.random.randint(0, height, number_of_pixels // 2)
    salt_x = np.random.randint(0, width, number_of_pixels // 2)
    noisy[salt_y, salt_x] = 255

    # Pepper noise
    pepper_y = np.random.randint(0, height, number_of_pixels // 2)
    pepper_x = np.random.randint(0, width, number_of_pixels // 2)
    noisy[pepper_y, pepper_x] = 0

    return noisy


# Calculate Mean Squared Error
def calculate_mse(original, result):
    original = original.astype(np.float32)
    result = result.astype(np.float32)

    mse = np.mean((original - result) ** 2)
    return round(float(mse), 2)


# Calculate PSNR
def calculate_psnr(mse):
    if mse == 0:
        return float("inf")

    return round(float(10 * np.log10((255 ** 2) / mse)), 2)


# Main image processing function
def process_image(image, add_noise, noise_amount):

    if image is None:
        return None, None, None, None, "Please upload an image."

    # Gradio provides an RGB NumPy image
    original = np.array(image).astype(np.uint8)

    # Add noise if selected
    if add_noise:
        working_image = add_salt_pepper_noise(
            original,
            float(noise_amount)
        )
    else:
        working_image = original.copy()

    # Apply Mean Filter
    mean_image = cv2.blur(
        working_image,
        (5, 5)
    )

    # Apply Gaussian Filter
    gaussian_image = cv2.GaussianBlur(
        working_image,
        (5, 5),
        0
    )

    # Apply Median Filter
    median_image = cv2.medianBlur(
        working_image,
        5
    )

    # Simple comparison values
    mean_mse = calculate_mse(original, mean_image)
    gaussian_mse = calculate_mse(original, gaussian_image)
    median_mse = calculate_mse(original, median_image)

    mean_psnr = calculate_psnr(mean_mse)
    gaussian_psnr = calculate_psnr(gaussian_mse)
    median_psnr = calculate_psnr(median_mse)

    comparison = (
        "Image Quality Comparison\n\n"
        f"Mean Filter     - MSE: {mean_mse} | PSNR: {mean_psnr} dB\n"
        f"Gaussian Filter - MSE: {gaussian_mse} | PSNR: {gaussian_psnr} dB\n"
        f"Median Filter   - MSE: {median_mse} | PSNR: {median_psnr} dB"
    )

    return (
        working_image,
        mean_image,
        gaussian_image,
        median_image,
        comparison
    )


# Gradio interface
with gr.Blocks() as demo:

    gr.Markdown(
        """
        # Image Smoothing Comparison Tool

        **Computer Vision Mini Project**

        This project compares Mean, Gaussian and Median
        filters for image smoothing and noise reduction.
        """
    )

    image_input = gr.Image(
        type="numpy",
        label="Upload Image"
    )

    add_noise = gr.Checkbox(
        label="Add Salt-and-Pepper Noise",
        value=False
    )

    noise_amount = gr.Slider(
        minimum=0.01,
        maximum=0.10,
        value=0.05,
        step=0.01,
        label="Noise Amount"
    )

    process_button = gr.Button("Apply Filters")

    gr.Markdown("### Results")

    with gr.Row():
        input_output = gr.Image(
            label="Original / Noisy Image"
        )

        mean_output = gr.Image(
            label="Mean Filter"
        )

    with gr.Row():
        gaussian_output = gr.Image(
            label="Gaussian Filter"
        )

        median_output = gr.Image(
            label="Median Filter"
        )

    comparison_output = gr.Textbox(
        label="Comparison"
    )

    process_button.click(
        fn=process_image,
        inputs=[
            image_input,
            add_noise,
            noise_amount
        ],
        outputs=[
            input_output,
            mean_output,
            gaussian_output,
            median_output,
            comparison_output
        ]
    )


# Run the application
if __name__ == "__main__":
    demo.launch()
