import numpy as np
import cv2

class Solution: 

  def __init__(self) -> None:
    pass
  def median_filter(self,image, kernel_size = 3):
    # Lấy kích thước của ảnh và kernel
    m, n = image.shape
    k = kernel_size // 2

    # Khởi tạo một mảng mới với kích thước bằng với ảnh gốc
    filtered_image = np.zeros_like(image)
    # Duyệt qua tất cả các pixel của ảnh
    for i in range(m):
        for j in range(n):
            # Lấy giá trị của các pixel trong kernel
            neighbors = []
            for x in range(-k, k+1):
                for y in range(-k, k+1):
                    # ví dụ x = 1, y = 2 => 
                    # Lấy vị trí của những ông bên trái phải trên dưới follow to i và j
                    # x , y = [ -1 0 1 ]
                    # 1 3 5 6 6
                    # 4 5 1 12 5
                    # 8 4 6 2 1
                    
                    # vi du filtered_image[i][j] = 1 => 
                    # neiber = [
                    #   3 5 6
                    #   5 1 12
                    #   4 6 2
                    # ] => sort to take middle = 1 2 3 4(replace 1 to 4) 5 5 6 12
                    # với x = 1 và y = 2 => append (0 , 1) , (0, 2) , (0,3), (...)
                    if i+x >= 0 and i+x < m and j+y >= 0 and j+y < n:
                      neighbors.append(image[i+x, j+y])

            # Sắp xếp các giá trị pixel trong kernel theo thứ tự tăng dần
            neighbors.sort()

            # Lấy giá trị median
            median = neighbors[len(neighbors) // 2]

            # Gán giá trị median cho pixel hiện tại
            filtered_image[i, j] = median

    return filtered_image
  def mean_filter(self, image, kernel_size = 3):
    # Thêm viền vào ảnh
    border = kernel_size // 2
    # tạo giá trị biên để khi tính sẽ k bị vượt ngoài gía trị
    padded_image = np.pad(image, pad_width=border, mode='constant', constant_values=0)
    # Khởi tạo kernel với tất cả giá trị bằng 1 và chia cho tổng số phần tử trong kernel
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    # Áp dụng kernel vào ảnh đã được đệm
    filtered_image = np.zeros_like(image)
    x , y = image.shape
    for i in range(x):
        for j in range(y):
          # nhân 2 ma trận 3 3 với ma trận kernel
          filtered_image[i, j] = np.sum(kernel * padded_image[i:i+kernel_size, j:j+kernel_size]) 
    else: print(filtered_image)
    return filtered_image
  def gaussian_kernel(self, size, sigma):
      kernel = np.zeros((size, size), dtype=np.float32)
      center = size // 2
      sum = 0.0
      for i in range(size):
          for j in range(size):
              x = i - center
              y = j - center
              # hằng số e của [i, j]
              kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
              # giá trị pixel [x,y] được tính bằng tổng trọng số của các pixel xung quanh nó nhân với giá trị kernel tương ứng
              # tổng trọng số pixel
              sum += kernel[i, j]
      # chia mỗi giá trị trong kernel cho tổng đó để đảm bảo tổng trọng số của kernel bằng 1.
      kernel = kernel / sum
      return kernel
  def gaussian_filter(self, image, kernel_size=3, sigma=1.0):
    # padding
    center_position = kernel_size // 2
    padded_image = cv2.copyMakeBorder(
      image, center_position, 
      center_position,
      center_position, 
      center_position,
      cv2.BORDER_REFLECT
    )
    kernel = self.gaussian_kernel(kernel_size, sigma)
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered_image[i, j] = np.sum(kernel * padded_image[i:i+kernel_size, j:j+kernel_size])
    return filtered_image
  def bilateral_filter(self, image, kernel_size = 3, signel = 1.0): 
    x, y = image.shape
    k = kernel_size // 2
    
    for i in range(x, y) :
      print()

    return

  