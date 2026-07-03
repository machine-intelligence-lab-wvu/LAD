'''

Authors: Zhang Ruihan
Github: https://github.com/zhangrh93/InvertibleCE?tab=readme-ov-file

'''

from PIL import Image

from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
            img = img.permute(1,2,0)
        return img,label

    def __len__(self):
        return len(self.imgs)
