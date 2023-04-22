import cv2
import noiser_filtering

class Main(noiser_filtering.Solution) :
  kernel_size = 3
  sigma = 1.0
  def __init__(self) -> None:
    pass
    
  # công thức sinh ra kernel


  def main(self):
    image = cv2.imread('datasets/salt_pepper.jpg', 0)
    # Đọc ảnh
    # Chuyển ảnh sang grayscale
    # Áp dụng median filter với kernel size = 3
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_img = self.mean_filter(image, 5)
    # Hiển thị ảnh gốc và ảnh đã xử lý
    cv2.imshow('Original Image', image)
    cv2.imshow('Filtered Image', filtered_img)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
  def gaussan_video(self):
      kernel_size = 5
      sigma = 1.0
      cap = cv2.VideoCapture('video_nhieu.mp4')

    # Lấy kích thước của video
      frame_width = int(cap.get(3))
      frame_height = int(cap.get(4))

    # Khởi tạo VideoWriter object để ghi video đã lọc
      out = cv2.VideoWriter('denoised_video.mp4',
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               cap.get(cv2.CAP_PROP_FPS),
                              (frame_width,frame_height))

    # Loop qua các frame của video đầu vào
      while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            filtered_image = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)

            out.write(filtered_image)
            # Hiển thị frame đã lọc
            cv2.imshow('Filtered Video', filtered_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      cap.release()
      out.release() 
      cv2.destroyAllWindows()

  def runWithVideo(self): 
     # Khởi tạo VideoCapture object với file video đầu vào
    cap = cv2.VideoCapture('video_nhieu.mp4')

    # Lấy kích thước của video
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Khởi tạo VideoWriter object để ghi video đã lọc
    out = cv2.VideoWriter('denoised_video.mp4',
                               cv2.VideoWriter_fourcc(*'mp4'),
                               cap.get(cv2.CAP_PROP_FPS),
                              (frame_width,frame_height))

    # Loop qua các frame của video đầu vào
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Áp dụng median filter với kernel size là 3
            filtered_frame = self.gaussian_filter(gray)
            # phục hồi màu sắc cho ảnh đã lọc bằng bilateral filtering
            bilateral_filtered = cv2.bilateralFilter(filtered_frame, 15, 75, 75)
            filtered_image = cv2.cvtColor(bilateral_filtered, cv2.COLOR_GRAY2BGR)
            filtered = cv2.cvtColor(filtered_frame, cv2.COLOR_GRAY2BGR)

            # kết hợp ảnh đã lọc và ảnh ban đầu để phục hồi màu sắc
            restored_image = cv2.addWeighted(frame, 0.5, filtered_image, 0.5, 0)
            # Ghi frame đã lọc vào file output
            out.write(filtered)

            # Hiển thị frame đã lọc
            cv2.imshow('Filtered Video', filtered)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # Giải phóng tài nguyên
    cap.release()
    out.release()
    cv2.destroyAllWindows()

Main().main()