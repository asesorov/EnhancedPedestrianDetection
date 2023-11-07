import albumentations as A

transform = A.Compose([
    A.CLAHE(always_apply=True, p=1.0, clip_limit=(1, 2), tile_grid_size=(2, 2)),
    A.RandomBrightness(always_apply=True, p=1.0, limit=(0.15, 0.15)),
    A.RandomContrast(always_apply=True, p=1.0, limit=(0.2, 0.2))
])
