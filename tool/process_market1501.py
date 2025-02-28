import os
import shutil

# Thư mục chứa ảnh Market-1501 (có thể là bounding_box_train hoặc bounding_box_test)
input_folder = "/home/micalab/AlignedReID/data/market1501_partial/query"
output_folder = "/home/micalab/AlignedReID/data/market1501_partial/track_query"

# Tạo thư mục đầu ra nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Duyệt qua tất cả các file ảnh
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):  # Kiểm tra định dạng ảnh
        person_id = filename.split("_")[0]  # Lấy ID người từ tên file
        
        # Bỏ qua ảnh có ID '-1' (thường là ảnh không xác định danh tính)
        if person_id == "-1":
            continue
        
        # Tạo thư mục tương ứng với ID nếu chưa có
        person_folder = os.path.join(output_folder, person_id)
        os.makedirs(person_folder, exist_ok=True)
        
        # Sao chép ảnh vào thư mục tương ứng
        src_path = os.path.join(input_folder, filename)
        dst_path = os.path.join(person_folder, filename)
        shutil.copy(src_path, dst_path)

print("✅ Dataset đã được chia thành các thư mục theo ID!")
