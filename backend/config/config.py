coco_config = """
train: "../data/train/images"  # Đường dẫn đến thư mục chứa ảnh huấn luyện
val: "../data/val/images"  # Đường dẫn đến thư mục chứa ảnh validation

nc: 9  # Số lượng class
names: ['motorbike', 'DHelmet', 'DNoHelmet', 'P1Helmet', 'P1NoHelmet', 'P2Helmet', 'P2NoHelmet', 'P0Helmet', 'P0NoHelmet']  # Danh sách tên class
"""
with open("config/coco.yaml", "w+", encoding="utf-8") as file:
    file.write(coco_config)

print("ok")
