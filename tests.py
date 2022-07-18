import cv2
import numpy as np

import datasets
import transforms


def check_dataset():
    dataset = datasets.WIDERFace('./data', 'train')
    idx = 0
    while True:
        print(f'{idx} / {len(dataset)}')
        image, label = dataset[idx]

        for i, bbox in enumerate(label['bbox']):
            print(bbox)
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

    cv2.destroyAllWindows()


def check_dataset2():
    T_compose = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
    ])
    dataset = datasets.WIDERFace('./data', 'train', T_compose)
    idx = 0
    while True:
        print(f'{idx} / {len(dataset)}')
        image, label = dataset[idx]

        image = image.copy()
        for i, bbox in enumerate(label['bbox']):
            cv2.rectangle(image, bbox.numpy().astype(int), (0, 255, 0), 1)

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

    cv2.destroyAllWindows()


def visualize_gt(gt, scale=255):
    views = []
    for view in gt:
        view = np.clip(view * scale, 0, 255)
        view = view.astype(np.uint8)
        view[:, 0] = 64
        views.append(view)
    views = np.concatenate(views, 1)
    return views


def test_encode_decode():
    T_compose = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
    ])

    dataset = datasets.WIDERFace('./data', 'val', T_compose)

    idx = 0

    while True:
        print(f'{idx} / {len(dataset)}')

        image, target = dataset[idx]
        image1 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image2 = image1.copy()

        encoded = transforms.label_encode(image, target)

        reg = visualize_gt(encoded[:4], scale=10)
        cv2.imshow('gt_reg', reg)
        cen = visualize_gt(encoded[4:5])
        cv2.imshow('gt_centerness', cen)

        cv2.imshow('image1', image1)
        cv2.imshow('image2', image2)
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

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # check_dataset()
    # check_dataset2()
    test_encode_decode()
