import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Load Image
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'pics/flower.png')
image = plt.imread(file_path)
print(image.shape)

# Image to Tensor
preprocess = transforms.ToTensor()
x = preprocess(image)
print(x.shape)

# Reverse to Img from Tensor
def reverse_to_img(x):
    x = x * 255
    x = x.clamp(0, 255)
    x = x.to(torch.uint8)
    to_pil = transforms.ToPILImage()
    return to_pil(x)

# Forward Diffusion Process
T = 1000
betas = torch.linspace(0.0001, 0.02, T)
imgs = []

for t in range(T):
    if t % 100 == 0:
        img = reverse_to_img(x)
        imgs.append(img)
    
    beta = betas[t]
    eps = torch.randn_like(x)
    x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * eps
    
"""
# Show imgs during Forward diffusion process
plt.figure(figsize=(15, 6))
for i, img in enumerate(imgs[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.title(f'Noise: {i * 100}')
    plt.axis('off')
plt.savefig('pics/forward_diffusion_process.png')
plt.show()
"""

"""
# Culmulative Product? -> torch.cumprod(): 지정된 축(차원)을 따라 누적 곱을 수행
# EX)
a = torch.tensor([1,2,3,4])
cumprod = torch.cumprod(a, dim=0)
# cumprod = tensor([ 1,  2,  6, 24])
"""

T = 1000
betas = torch.linspace(0.0001, 0.02, T)

# q(x_t|x_0) sampling -> x0으로부터 한번에 xt번째의 이미지를 불러온다
def add_noise(x0, t, betas):
    T = len(betas)
    assert t >= 1 and t <= T

    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    t_idx = t-1
    a_t_bar = alpha_bars[t_idx]

    eps = torch.randn_like(x0)
    y = torch.sqrt(a_t_bar) * x0 + torch.sqrt(1-a_t_bar) * eps

    return y

t = 100
x_t = add_noise(x, t, betas)

img = reverse_to_img(x_t)
plt.imshow(img)
plt.title(f'Noise: {t}, directly sampling from x0')
plt.axis('off')
plt.savefig('pics/q_xt_x0.png')
plt.show()