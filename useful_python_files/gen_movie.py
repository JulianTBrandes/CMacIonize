import cv2
import os

def pngs_to_mp4(input, output_file, fps=30):
    input_folder = "/sharedscratch/jb450/CMacIonize/CMacIonize/SW_Test/Png/"+input+"/"
    images = sorted(
        [img for img in os.listdir(input_folder) if img.endswith(".png")]
    )

    if not images:
        raise ValueError("No PNG files found.")

    first_frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(input_folder, image))
        video.write(frame)

    video.release()
    print("Video saved to:", output_file)

pngs_to_mp4("2D", "2D.mp4", fps=24)