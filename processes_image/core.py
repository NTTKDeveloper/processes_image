import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Thư viện để hiển thị thanh tiến trình
import time 

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
def edge_detection_to_vector(image, threshold1=50, threshold2=150, kmedian_blurred=1, kbox_blurred=(1, 1)):
    start_time = time.time()  # Bắt đầu đo thời gian
    
    # Chuyển đổi ảnh sang thang độ xám
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Làm mờ bằng bộ lọc trung vị
    median_blurred = cv2.medianBlur(image, kmedian_blurred)

    # Làm mờ Gaussian
    # gaussian_blurred = cv2.GaussianBlur(median_blurred, kgaussian_blurred, sigma)
    blurred = cv2.blur(median_blurred, kbox_blurred)

    # Phát hiện cạnh bằng Canny
    edges = cv2.Canny(blurred, threshold1, threshold2)

    # Tìm các điểm cạnh
    points = np.argwhere(edges > 0)
    points = points[:, [1, 0]]  # Đổi thứ tự x, y

    # Tính vector gradient
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    vectors = np.column_stack((grad_x[points[:, 1], points[:, 0]], grad_y[points[:, 1], points[:, 0]]))

    end_time = time.time()  # Kết thúc đo thời gian
    execution_time_ms = (end_time - start_time) * 1000  # Đổi sang mili giây

    return points, vectors, execution_time_ms

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

#Cửa sổ trượt để xử lí hình ảnh
def sliding_window(image, size=(28, 28), position=(0, 0)):
    """
    Cắt một ảnh nhỏ từ ảnh gốc, nếu vượt ngoài phạm vi sẽ thêm padding màu đen.

    Args:
        image (ndarray): Ảnh đầu vào (ảnh gốc).
        size (tuple): Kích thước ảnh cần cắt (chiều rộng, chiều cao).
        position (tuple): Tọa độ góc trên bên trái (x, y) của khung cắt.

    Returns:
        ndarray: Ảnh đã cắt (bao gồm padding nếu cần).
    """
    width, height = size
    x, y = position

    # Kích thước ảnh gốc
    img_h, img_w, _ = image.shape

    # Xác định biên trong ảnh gốc
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(img_w, x + width)
    y_end = min(img_h, y + height)

    # Cắt phần nằm trong ảnh
    cropped_part = image[y_start:y_end, x_start:x_end]

    # Tính toán kích thước phần cần padding
    pad_top = max(0, -y)  # Padding trên
    pad_left = max(0, -x)  # Padding trái
    pad_bottom = max(0, (y + height) - img_h)  # Padding dưới
    pad_right = max(0, (x + width) - img_w)  # Padding phải

    # Thêm padding bằng màu đen
    cropped_with_padding = cv2.copyMakeBorder(
        cropped_part,
        pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # Màu đen
    )

    return cropped_with_padding