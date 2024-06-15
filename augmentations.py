'''
Defines easy, medium and hard levels of augmentations
'''

import albumentations as A

def get_augmentations(image_w, image_h, mode='easy'):
    easy_augs = [
        A.Resize(image_w, image_h),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]

    medium_augs = [
        A.RandomSizedCrop(min_max_height=(image_h // 2, image_h), w2h_ratio=1, height=image_h, width=image_w, p=0.5),
        A.CLAHE(p=0.8),
        A.RandomBrightnessContrast(p=0.8),
        A.RandomGamma(p=0.8)
    ]

    hard_augs = [A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5)
    ], p=0.5)]


    if mode == 'easy':
        augs = easy_augs

    elif mode == 'medium':
        augs = [*easy_augs, *medium_augs]

    elif mode == 'hard':
        augs = [*easy_augs, *hard_augs, *medium_augs]

    else:
        return None
    
    return A.Compose(augs, additional_targets={'mask_roof': 'mask', 'mask_height': 'mask'})


def get_tta_augs():
    augs = [
        A.CLAHE(p=0.7),
        A.RandomBrightnessContrast(p=0.7),
        A.RandomGamma(p=0.7)
    ]


    return A.Compose(augs)
