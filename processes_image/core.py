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
def edge_detection_to_vector(image, threshold1=50, threshold2=150, kmedian_blurred=1, kgaussian_blurred=(1,1), sigma=0):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    median_blurred = cv2.medianBlur(image, kmedian_blurred)  # Kernel size = 5

    # Làm mờ Gaussian làm mờ mà vẫn giữ chi tiết 
    gaussian_blurred = cv2.GaussianBlur(median_blurred, kgaussian_blurred, sigma)  # Kernel size (5x5) và sigma = 0

    # Phát hiện cạnh bằng Canny
    edges = cv2.Canny(gaussian_blurred, threshold1, threshold2)

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


def resize_edge_vectors(points, vectors, original_size, new_size):
    """
    Resize các vector cạnh và điểm sao cho phù hợp với kích thước ảnh mới.
    
    Parameters:
    - points: Danh sách tọa độ điểm cạnh (numpy array).
    - vectors: Danh sách vector gradient (numpy array).
    - original_size: Kích thước gốc của ảnh (height, width).
    - new_size: Kích thước mới của ảnh (height, width).

    Returns:
    - resized_points: Điểm cạnh đã được resize.
    - resized_vectors: Vector gradient đã được resize.
    """
    # Tính tỷ lệ thay đổi kích thước
    scale_x = new_size[1] / original_size[1]  # Tỷ lệ theo chiều ngang
    scale_y = new_size[0] / original_size[0]  # Tỷ lệ theo chiều dọc
    
    # Resize điểm cạnh
    resized_points = points.copy()
    resized_points[:, 0] = points[:, 0] * scale_x  # Điều chỉnh tọa độ x
    resized_points[:, 1] = points[:, 1] * scale_y  # Điều chỉnh tọa độ y
    
    # Resize vector gradient
    resized_vectors = vectors.copy()
    resized_vectors[:, 0] *= scale_x  # Điều chỉnh độ dài vector theo x
    resized_vectors[:, 1] *= scale_y  # Điều chỉnh độ dài vector theo y

    return resized_points, resized_vectors