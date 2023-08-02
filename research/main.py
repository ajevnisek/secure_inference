import pickle

import detectors
import timm
import torch.nn
import torchvision.transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm

activation = {}
def getActivation(name):
    # the hook signat`ure
    def hook(model, input, output):
        if name not in activation:
            activation[name] = [(output[0].detach() >= 0).to(torch.uint8)]
        else:
            activation[name].append((output[0].detach() >= 0).to(torch.uint8))
    return hook


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def main():
    model = timm.create_model("resnet18_cifar100", pretrained=True)

    transforms = Compose([ToTensor(),
                          Normalize((0.5071, 0.4867, 0.4408),
                                    (0.2675, 0.2565, 0.2761))])
    test_dataset = CIFAR100(root='data/', train=False, download=False, transform=transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    images, labels = next(iter(test_dataloader))
    model.act1.register_forward_hook(getActivation('act1'))
    model.layer4[1].act1.register_forward_hook(getActivation('layer4_1_act1'))
    correct, total = 0, 0
    pbar = tqdm(test_dataloader)
    with torch.no_grad():
        for batch in pbar:
            images, labels = batch
            pred = model(images)
            correct += (pred.argmax(1) == labels).sum().item()
            total += pred.shape[0]
            pbar.set_description(f"acc: {correct / total * 100:.2f}[%]")
    print(f"Accuracy: {correct / total * 100.0:.2f} [%]")
    with open('activation.pkl', 'wb') as f:
        pickle.dump(activation, f)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
