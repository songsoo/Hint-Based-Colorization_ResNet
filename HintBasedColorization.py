import torch
from torch.autograd import Variable
from torchvision import transforms
import cv2
import random
import numpy as np
import os
from torchvision import transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
import tqdm
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import mse_loss as mse



class ColorHintTransform(object):
  def __init__(self, size=256, mode="training"):
    super(ColorHintTransform, self).__init__()
    self.size = size
    self.mode = mode
    self.transform = transforms.Compose([transforms.ToTensor()])

  def bgr_to_lab(self, img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, ab = lab[:, :, 0], lab[:, :, 1:]
    return l, ab

  def hint_mask(self, bgr, threshold=[0.95, 0.97, 0.99]):
    h, w, c = bgr.shape
    mask_threshold = random.choice(threshold) # 3 threshold random choice
    mask = np.random.random([h, w, 1]) > mask_threshold # Create a mask with only hint values
    return mask

  def img_to_mask(self, mask_img):
    mask = mask_img[:, :, 0, np.newaxis] >= 255
    return mask

  def __call__(self, img, mask_img=None):
    threshold = [0.95, 0.97, 0.99]
    if (self.mode == "training") | (self.mode == "validation"):
      image = cv2.resize(img, (self.size, self.size))
      mask = self.hint_mask(image, threshold)

      hint_image = image * mask # hint_image we know

      l, ab = self.bgr_to_lab(image) # split image into l and ab
      l_hint, ab_hint = self.bgr_to_lab(hint_image) # split hint_image into l and ab
      return self.transform(l), self.transform(ab), self.transform(ab_hint), self.transform(mask) # l, ab, ab_hint, mask transform apply # Add mask


    elif self.mode == "testing":
      image = cv2.resize(img, (self.size, self.size))
      mask = self.img_to_mask(mask_img)
      hint_image = image * self.img_to_mask(mask_img) # Changing the image itself to hint_image

      l, _ = self.bgr_to_lab(image)
      _, ab_hint = self.bgr_to_lab(hint_image)

      return self.transform(l), self.transform(ab_hint), self.transform(mask)

    else:
      return NotImplementedError

class ColorHintDataset(data.Dataset):
  def __init__(self, root_path, size, mode="train"):
      super(ColorHintDataset, self).__init__()

      self.root_path = root_path
      self.size = size
      self.transforms = None
      self.examples = None
      self.hint = None
      self.mask = None

  def set_mode(self, mode):
      self.mode = mode
      self.transforms = ColorHintTransform(self.size, mode)
      if mode == "training":
        train_dir = os.path.join(self.root_path, "train")
        self.examples = [os.path.join(self.root_path, "train", dirs) for dirs in os.listdir(train_dir)]
      elif mode == "validation":
        val_dir = os.path.join(self.root_path, "val")
        self.examples = [os.path.join(self.root_path, "val", dirs) for dirs in os.listdir(val_dir)]
      elif mode == "testing":
        hint_dir = os.path.join(self.root_path, "hint")
        mask_dir = os.path.join(self.root_path, "mask")
        self.hint = [os.path.join(self.root_path, "hint", dirs) for dirs in os.listdir(hint_dir)]
        self.mask = [os.path.join(self.root_path, "mask", dirs) for dirs in os.listdir(mask_dir)]
      else:
        raise NotImplementedError

  def __len__(self):
      if self.mode != "testing":
        return len(self.examples)
      else:
        return len(self.hint)

  def __getitem__(self, idx):
    if self.mode == "testing":
      hint_file_name = self.hint[idx]
      mask_file_name = self.mask[idx]
      hint_img = cv2.imread(hint_file_name)
      mask_img = cv2.imread(mask_file_name)  # Add mask

      input_l, input_hint, input_mask = self.transforms(hint_img, mask_img)
      sample = {"l": input_l, "hint": input_hint, "mask": input_mask,
                "file_name": "image_%06d.png" % int(os.path.basename(hint_file_name).split('.')[0])}  # Add mask

    else:
      file_name = self.examples[idx]
      img = cv2.imread(file_name)
      l, ab, hint, mask = self.transforms(img)  # Add mask
      sample = {"l": l, "ab": ab, "hint": hint, "mask": mask}  # Add mask

    return sample

def tensor2im(input_image, imtype=np.uint8):  # Tensor type -> image type
  if isinstance(input_image, torch.Tensor):
    image_tensor = input_image.data
  else:
    return input_image
  image_numpy = image_tensor[0].cpu().float().numpy()
  if image_numpy.shape[0] == 1:
    image_numpy = np.tile(image_numpy, (3, 1, 1))
  image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0))), 0, 1) * 255.0
  return image_numpy.astype(imtype)


class UnetGenerator(nn.Module):
  def __init__(self, norm_layer=nn.BatchNorm2d):
    super(UnetGenerator, self).__init__()

    self.model_down1 = nn.Sequential(
      nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True),  # l, a, b, mask --> 4 input channels
      norm_layer(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
      nn.BatchNorm2d(64),
    )

    self.model_down2 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
      norm_layer(128),
      nn.ReLU(),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
      nn.BatchNorm2d(128),
    )

    self.model_down3 = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
      norm_layer(256),
      nn.ReLU(),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
      nn.BatchNorm2d(256),
    )

    self.model_down4 = nn.Sequential(
      nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
      norm_layer(512),
      nn.ReLU(),
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
      nn.BatchNorm2d(512),
    )

    self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    self.model_bridge = nn.Sequential(
      nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=True),
      norm_layer(1024),
      nn.ReLU(),
      nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1, bias=True),
      nn.BatchNorm2d(1024),
     )

    self.model_trans1 = nn.Sequential(
     nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
     nn.BatchNorm2d(512),
     nn.ReLU(),
     )

    self.model_up1 = nn.Sequential(
      nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(),
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(512),
    )

    self.model_trans2 = nn.Sequential(
      nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),
    )

    self.model_up2 = nn.Sequential(
      nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(256),
    )

    self.model_trans3 = nn.Sequential(
      nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
    )

    self.model_up3 = nn.Sequential(
      nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
    )

    self.model_trans4 = nn.Sequential(
      nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
      nn.ReLU(),
      nn.ReLU(),
    )

    self.model_up4 = nn.Sequential(
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(64),
    )

    self.model_out = nn.Sequential(
      nn.Conv2d(64, 3, 3, 1, 1),  # l, a, b --> 3 output channels
      nn.Tanh(),
    )

  def forward(self, input_lab):
    down1 = self.model_down1(input_lab)
    pool1 = self.pool(down1)
    down2 = self.model_down2(pool1)
    pool2 = self.pool(down2)
    down3 = self.model_down3(pool2)
    pool3 = self.pool(down3)
    down4 = self.model_down4(pool3)
    pool4 = self.pool(down4)

    bridge = self.model_bridge(pool4)

    trans1 = self.model_trans1(bridge)


    concat1 = torch.cat([trans1, down4], dim=1)
    up1 = self.model_up1(concat1)
    trans2 = self.model_trans2(up1)
    concat2 = torch.cat([trans2, down3], dim=1)
    up2 = self.model_up2(concat2)
    trans3 = self.model_trans3(up2)
    concat3 = torch.cat([trans3, down2], dim=1)

    up3 = self.model_up3(concat3)
    print(up3.shape)

    trans4 = self.model_trans4(up3)
    concat4 = torch.cat([trans4, down1], dim=1)
    up4 = self.model_up4(concat4)

    return self.model_out(up4)

def tensor2im(input_image, imtype=np.uint8):
  if isinstance(input_image, torch.Tensor):
    image_tensor = input_image.data
  else:
    return input_image
  image_numpy = image_tensor[0].cpu().float().numpy()
  if image_numpy.shape[0] == 1:
    image_numpy = np.tile(image_numpy, (3, 1, 1))
  image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0))), 0, 1) * 255.0
  return image_numpy.astype(imtype)

# Change to your data root directory
root_path = "data/"
# Depend on runtime setting
use_cuda = True
os.environ['KMP_DUPLICATE_LIB_OK']='True'

train_dataset = ColorHintDataset(root_path, 256,"training")
train_dataset.set_mode("training")
train_dataloader = data.DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataset = ColorHintDataset(root_path, 256,"validation")
val_dataset.set_mode("validation")
val_dataloader = data.DataLoader(val_dataset, batch_size=4, shuffle=True)

model = UnetGenerator()  # load model

class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


# ================== define train and validation ==================

def train(model, train_dataloader, optimizer, criterion, epoch):
  print('[Training] epoch {} '.format(epoch))
  model.train()
  losses = AverageMeter()

  for i, data in enumerate(train_dataloader):

    # if use_cuda:
    l = data["l"].cuda()
    ab = data["ab"].cuda()
    hint = data["hint"].cuda()
    mask = data["mask"].cuda()

    # concat
    gt_image = torch.cat((l, ab), dim=1).cuda()
    # print('\n===== img size =====\n', gt_image.shape)
    hint_image = torch.cat((l, hint, mask), dim=1).cuda()
    # print('===== hint size =====\n', hint_image.shape)

    # run forward
    output_ab = model(hint_image)
    loss = criterion(output_ab, gt_image)
    losses.update(loss.item(), hint_image.size(0))

    # compute gradient and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
      print('Train Epoch : [{}] [{} / {}]\tLoss{loss.val:.4f}'.format(epoch, i, len(train_dataloader), loss=losses))


def validation(model, train_dataloader, criterion, epoch):
  model.eval()
  losses = AverageMeter()

  for i, data in enumerate(val_dataloader):

    # if use_cuda:
    l = data["l"].cuda()
    ab = data["ab"].cuda()
    hint = data["hint"].cuda()
    mask = data["mask"].cuda()

    # concat
    gt_image = torch.cat((l, ab), dim=1).cuda()
    # print('\n===== img size =====\n', gt_image.shape)
    hint_image = torch.cat((l, hint, mask), dim=1).cuda()
    # print('===== hint size =====\n', hint_image.shape)

    # run model and store loss
    output_ab = model(hint_image)
    loss = criterion(output_ab, gt_image)
    losses.update(loss.item(), hint_image.size(0))

    gt_np = tensor2im(gt_image)
    # print('\n===== gt size =====\n', gt_np.shape)
    hint_np = tensor2im(output_ab)
    # print('===== hint size =====\n', hint_np.shape)

    gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_LAB2BGR)
    hint_bgr = cv2.cvtColor(hint_np, cv2.COLOR_LAB2BGR)

    os.makedirs('data/ground_truth', exist_ok=True)
    cv2.imwrite('data/ground_truth/gt_' + str(i) + '.jpg', gt_bgr)

    os.makedirs('data/predictions', exist_ok=True)
    cv2.imwrite('data/predictions/pred_' + str(i) + '.jpg', hint_bgr)

    if i % 100 == 0:
      print('Validation Epoch : [{} / {}]\tLoss{loss.val:.4f}'.format(i, len(val_dataloader), loss=losses))

      cv2.imshow('',gt_bgr)
      cv2.imshow('',hint_bgr)

  return losses.avg


# ================== define psnr and psnr_loss ==================

def psnr(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
  if not isinstance(input, torch.Tensor):
    raise TypeError(f"Expected torch.Tensor but got {type(target)}.")

  if not isinstance(target, torch.Tensor):
    raise TypeError(f"Expected torch.Tensor but got {type(input)}.")

  if input.shape != target.shape:
    raise TypeError(f"Expected tensors of equal shapes, but got {input.shape} and {target.shape}")

  return 10. * torch.log10(max_val ** 2 / mse(input, target, reduction='mean'))


def psnr_loss(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
  return -1. * psnr(input, target, max_val)

class PSNRLoss(nn.Module):
  def __init__(self, max_val: float) -> None:
    super(PSNRLoss, self).__init__()
    self.max_val: float = max_val

  def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return psnr_loss(input, target, self.max_val)


model = UnetGenerator()
criterion = PSNRLoss(2.)

optimizer = optim.Adam(model.parameters(), lr=0.00025)  # 1e-2 # 0.0005 # 0.00025 # 0.0002
epochs = 5
best_losses = 10

save_path = 'data/result'
os.makedirs(save_path, exist_ok=True)
output_path = os.path.join(save_path, 'validation_model.tar')

model.cuda()

for epoch in range(epochs):
  train(model, train_dataloader, optimizer, criterion, epoch)
  with torch.no_grad():
    val_losses = validation(model, val_dataloader, criterion, epoch)

  if best_losses > val_losses:
    best_losses = val_losses
    torch.save(model.state_dict(),
               'data/PSNR/PSNR-epoch-{}-losses-{:.5f}.pth'.format(epoch + 1, best_losses))


def image_save(img, path):
  if isinstance(img, torch.Tensor):
    img = np.asarray(transforms.ToPILImage()(img))
  img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
  cv2.imwrite(path, img)


result_save_path = "data/result"  # input best loss's model

test_dataset = ColorHintDataset(root_path, 256)
test_dataset.set_mode('testing')
print('Test length : ', len(test_dataset))

test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

model = UnetGenerator().cuda()

model_path = os.path.join('data/PSNR/PSNR-epoch-2-losses--40.05382.pth')
model.load_state_dict(torch.load(model_path))


# =========================================================

def test(model, test_dataloader):
  model.eval()  # same as testing mode
  for i, data in enumerate(test_dataloader):
    l = data["l"].cuda()
    # print('\n===== l size =====\n', l.shape) # [1, 1, 128, 128]
    hint = data["hint"].cuda()
    # print('\n===== hint size =====\n', hint.shape) # [1, 2, 128, 128]
    mask = data["mask"].cuda()  # add mask

    file_name = data['file_name']

    with torch.no_grad():
      out = torch.cat((l, hint, mask), dim=1)  # add mask
      pred_image = model(out)

      for idx in range(len(file_name)):
        image_save(pred_image[idx], os.path.join(result_save_path, file_name[idx]))
        print(file_name[idx])


# =========================================================

test(model, test_dataloader)