import albumentations as alb
import albumentations.augmentations.functional as albF
import albumentations.augmentations.transforms as albT

def transforms():
    transforms=alb.Compose(
            [
            alb.OneOf(
             [
             albT.Blur(),
             albT.RandomBrightness(),
             alb.RandomBrightnessContrast(contrast_limit=0.5)
             ], p=0.5),

            alb.OneOf(
            [
             alb.Rotate(limit=10),
            ], p=0.5),

            

            albT.RandomSunFlare((0,0,1,1),src_radius=75,p=0.5,num_flare_circles_upper=3,num_flare_circles_lower=1, src_color=(150, 150, 150)),
            alb.ShiftScaleRotate(scale_limit=0.2,rotate_limit=0,shift_limit=0,p=0.5),        
            ],
        #     additional_targets={'image0': 'image', 'image1': 'image','image2': 'image'}
        )
    return transforms