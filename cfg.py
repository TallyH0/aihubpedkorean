import albumentations as A

batch_size = 502
imgh = 128
imgw = 64
num_class = 502
device = 'cuda'

max_epoch = 200
lr = 1e-2

model = 'mit_b0'
dir_image = 'F:/Data/pedestrain_korean/image/Training/data'
dir_test = 'F:/Data/pedestrain_korean/image/Validation/data'
save_path = 'aihub_korea_ped_reid_training_set_mit_b0.pth'

augmentation = A.Compose([
    A.HorizontalFlip(),
    A.CoarseDropout(max_holes=2, max_height=20, max_width=10),
    # A.ColorJitter()
])