import cv2
import torch
import torchvision

import tests
import models
import datasets
import transforms


class Module:
    def __init__(self, model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = models.Detector()

        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model'])
        net = net.to(device)
        net = net.eval()

        T_compose = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])

        self.device = device
        self.net = net
        self.transform = T_compose

    @torch.inference_mode()
    def __call__(self, image):
        x = self.transform(image).unsqueeze(0)
        x = x.to(self.device)
        out = self.net(x)
        print(out.shape)
        out[:, 4:] = torch.sigmoid(out[:, 4:])
        print(out[:, 4:].max(), out[:, 4:].min())
        out = out.cpu().numpy().squeeze()
        return out


def visualize_eval(
    model_path='./logs/widerface/fcos2/models/model_050.pt',
    size=(512, 512),
):
    module = Module(model_path)

    dataset = datasets.WIDERFace(
        './data', 'val', transforms.Resize(size))

    idx = 0
    while True:
        image, label = dataset[idx]

        out = module(image)

        reg = tests.visualize_gt(out[:4], scale=10)
        cen = tests.visualize_gt(out[4:5], scale=255)
        cv2.imshow('reg', reg)
        cv2.imshow('cen', cen)

        decoded = transforms.label_decode(out)

        for bbox in label['bbox']:
            cv2.rectangle(image, bbox.astype(int), (0, 255, 0), 1)
        for t in decoded:
            bbox = list(map(int, t['bbox']))
            cv2.rectangle(image, bbox, (255, 0, 255), 1)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('image', image)
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
    visualize_eval()
