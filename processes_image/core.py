import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Thư viện để hiển thị thanh tiến trình

# Hàm căn chỉnh vector vào trung tâm ảnh
def center_vectors(points, vectors, image_size):
    # Tính trung tâm của các điểm cạnh
    center_of_points = points.mean(axis=0)

    # Tính trung tâm của ảnh
    center_of_image = np.array([image_size[1] / 2, image_size[0] / 2])

    # Tịnh tiến các điểm để đưa khối tâm về trung tâm ảnh
    translation = center_of_image - center_of_points
    centered_points = points + translation

    return centered_points, vectors

# Tích vô hướng vector
def calculate_total_dot_product(vectors):
    total_dot_product = np.sum(np.dot(vectors, vectors.T)) - np.sum(np.linalg.norm(vectors, axis=1)**2)
    return total_dot_product / 2  # Vì tích vô hướng được tính cho các cặp duy nhất

# Tích vô hướng trung bình
def calculate_avg_dot_product(vectors):
    num_vectors = len(vectors)
    num_pairs = num_vectors * (num_vectors - 1) / 2
    if num_pairs == 0:
        return 0
    total_dot_product = calculate_total_dot_product(vectors)
    return total_dot_product / num_pairs

# Hàm phát hiện cạnh và chuyển thành vector
def edge_detection_to_vector(image, threshold1, threshold2):
    # Phát hiện cạnh bằng Canny
    edges = cv2.Canny(image, threshold1, threshold2)

    # Tìm các điểm cạnh
    points = np.argwhere(edges > 0)
    points = points[:, [1, 0]]

    # Tính vector gradient
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    vectors = np.column_stack((grad_x[points[:, 1], points[:, 0]], grad_y[points[:, 1], points[:, 0]]))
    return points, vectors

# Chuẩn hóa vector
def normalize_vectors(vectors):
    magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
    magnitudes[magnitudes == 0] = 1  # Tránh chia cho 0
    return vectors / magnitudes

# Tính mật độ cạnh
def calculate_edge_density(edges):
    return np.sum(edges > 0) / edges.size

# Histogram góc
def calculate_angle_histogram(vectors, bins=36):
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    angles = np.degrees(angles)
    angles[angles < 0] += 360
    hist, _ = np.histogram(angles, bins=bins, range=(0, 360))
    return hist

# Hàm tái tạo ảnh từ vector
def vector_to_image(points, image_shape):
    reconstructed_image = np.zeros(image_shape, dtype=np.uint8)
    for point in points:
        x, y = point
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            reconstructed_image[y, x] = 255
    return reconstructed_image