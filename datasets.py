import cv2
import numpy as np
import torchvision


class WIDERFace(torchvision.datasets.WIDERFace):
    def __init__(self, root, split='train', transform=None):
        super().__init__(root, split)
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.getdata(idx)
        if self.transform:
            image, label = self.transform(image, label)
        return image, label

    def getdata(self, idx):
        image, label = super().__getitem__(idx)
        image = np.asarray(image)
        return image, label


def check_dataset():
    dataset = WIDERFace('./data', 'train')
    idx = 0
    while True:
        print(f'{idx} / {len(dataset)}')
        image, label = dataset[idx]

        for i, bbox in enumerate(label['bbox']):
            blur = label['blur'][i]
            expression = label['expression'][i]
            illumination = label['illumination'][i]
            occlusion = label['occlusion'][i]
            pose = label['pose'][i]
            invalid = label['invalid'][i]
            code = f'{blur}{expression}{illumination}{occlusion}{pose}{invalid}'
            cv2.putText(image, code, bbox.numpy()[:2], 1, 1, (0, 255, 0), 1)
            cv2.rectangle(image, bbox.numpy(), (0, 255, 0), 1)

        cv2.imshow('image', cv2.cvtColor(image, 4))
        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == ord('q'):
            idx -= 1
        else:
            idx += 1

        if idx < 0:
            idx = len(dataset)-1
        elif idx >= len(dataset):
            idx = 0


if __name__ == '__main__':
    check_dataset()
