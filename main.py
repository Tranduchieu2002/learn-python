import cv2
import numpy as np

def chuvi() :
  r = float(input());

  return round(r * r *  4 * pi );
def canbac3() :
  arr = [1,2,4,6 ]
  arr.extend([2,3]);
  for idx in range(len(arr)): 
    print(arr[idx]);

class Solution :
  mot_mang = [[ 1, 6 ,7 ,1 ,2] , [ 21 ,6 ,6 ,1, 8] ]
  def __init__(self) -> None:
    pass
  
  def median_filter(self,image, kernel_size):
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
                    # Lấy vị trí của những ông bên trái phải trên dưới follow to i và j
                    # 3 5 6
                    # 4 5 1
                    #
                    if i+x >= 0 and i+x < m and j+y >= 0 and j+y < n:
                        neighbors.append(image[i+x, j+y])

            # Sắp xếp các giá trị pixel trong kernel theo thứ tự tăng dần
            neighbors.sort()

            # Lấy giá trị median
            median = neighbors[len(neighbors) // 2]

            # Gán giá trị median cho pixel hiện tại
            filtered_image[i, j] = median

    return filtered_image
  def main(self):
    image = cv2.imread('salt_pepper.jpg', 0);
    # Đọc ảnh
    print(image.shape)
    # Chuyển ảnh sang grayscale
    # Áp dụng median filter với kernel size = 3
    filtered_img = self.median_filter(image, 3)
    gray_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Hiển thị ảnh gốc và ảnh đã xử lý
    cv2.imshow('Original Image', gray_img)
    cv2.imshow('Filtered Image', filtered_img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

Solution().main()