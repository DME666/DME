import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import os
from torchvision import transforms
from models.MambaVision import MambaVision



# -------------------- Grad-CAM 计算 --------------------


def generate_gradcam(model, image_tensor, target_layer, device):
    model.eval()

    gradients = []
    activations = []

    # 后向钩子需要接收模块、输入梯度和输出梯度
    def backward_hook(module, grad_input, grad_output):
        # 保存梯度输出（通常grad_output是一个元组，取第一个元素）
        gradients.append(grad_output[0].detach())  # 确保梯度被正确捕获

    def forward_hook(module, input, output):
        # 保存前向传播的激活值（如果输出是元组，取第一个元素）
        activations.append(output[0].detach() if isinstance(output, tuple) else output.detach())

    # 确保模型和输入张量在正确的设备上
    model.to(device)
    image_tensor = image_tensor.to(device).requires_grad_(True)  # 确保输入需要梯度

    # 注册钩子
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)

    # 前向传播
    output = model(image_tensor)['logits']
    target_class = output.argmax(dim=1).item()

    # 反向传播：确保保留计算图
    model.zero_grad()
    output[0, target_class].backward(retain_graph=True)  # 使用retain_graph防止计算图被释放

    # 移除钩子
    handle_forward.remove()
    handle_backward.remove()

    # 检查梯度和激活是否被正确捕获
    if not gradients or not activations:
        raise ValueError("未捕获到梯度或激活，请检查目标层是否正确。")

    grad = gradients[0]
    act = activations[0]

    # 处理梯度维度：假设激活是4D (batch, channels, H, W)
    if grad.dim() == 3:  # 一维卷积可能输出3D张量 (batch, channels, length)
        pooled_grad = torch.mean(grad, dim=[0, 2], keepdim=True)  # 平均空间维度
    elif grad.dim() == 4:
        pooled_grad = torch.mean(grad, dim=[0, 2, 3], keepdim=True)
    else:
        raise RuntimeError(f"不支持的梯度维度: {grad.shape}")

    # 计算CAM
    cam = torch.sum(pooled_grad * act, dim=1).squeeze()
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # 避免除以零

    return cam.cpu().numpy()




# -------------------- 可视化并保存 --------------------

def overlay_heatmap(image_path, cam):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    return overlay


def create_white_background_image(image, height, width):
    """创建白色背景并将图像放在上面，确保图像大小适配目标背景"""
    # 调整热力图大小到指定的背景尺寸
    image_resized = cv2.resize(image, (width, height))

    # 创建白色背景
    white_background = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 将调整后的图像放置到白色背景中间
    y_offset = (white_background.shape[0] - image_resized.shape[0]) // 2
    x_offset = (white_background.shape[1] - image_resized.shape[1]) // 2
    white_background[y_offset:y_offset + image_resized.shape[0],
    x_offset:x_offset + image_resized.shape[1]] = image_resized

    return white_background


def save_concat_image_with_labels(images, labels, output_path):
    """拼接 6 张图像并保存，并且在每张下面加上标签"""
    # 定义每个图像的高度和宽度
    image_height, image_width = images[0].shape[:2]

    # 创建一个白色背景的拼接图像
    total_width = sum([img.shape[1] for img in images])
    max_height = max([img.shape[0] for img in images])

    concat_image = np.ones((max_height + 30, total_width, 3), dtype=np.uint8) * 255  # 额外空间用于标签
    x_offset = 0

    for i, image in enumerate(images):
        y_offset = (max_height - image.shape[0]) // 2
        concat_image[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image
        x_offset += image.shape[1]

        # 添加标签
        cv2.putText(concat_image, labels[i], (x_offset - image.shape[1] // 2, max_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite(output_path, cv2.cvtColor(concat_image, cv2.COLOR_RGB2BGR))
    print(f"Grad-CAM heatmap with labels saved at: {output_path}")

    # 显示图片
    plt.figure(figsize=(20, 5))
    plt.imshow(concat_image)
    plt.axis("off")
    plt.show()


# -------------------- 运行热力图分析 --------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有可用的 GPU

    # image_path = os.path.join(IMG_CUT_COVER, "0000-0052L_1001.jpg")
    # image_path = os.path.join(IMG_CUT_COVER, "0000-0108L_1026.jpg")
    # image_path = os.path.join(IMG_CUT_COVER, "0000-0137L_1004.jpg")
    # image_path = os.path.join(IMG_CUT_COVER, "0000-0865R_1000.jpg")
    # image_path = os.path.join(IMG_CUT_COVER, "0000-1362L_1004.jpg")
    # image_path = os.path.join(IMG_CUT_COVER, "0000-1362L_1001.jpg")
    image_path = '/home/zhiqinkun/DME/img/train/0/0000-0034L_1003.jpg'
    # model_paths = '/home/zhiqinkun/project/DME/ManbaVision/log/纯图片/100ep5_2/mambavision-T-1K minlr=3e-7/best_model_epoch_3.pth'
    model_paths = '/home/zhiqinkun/project/DME/ManbaVision/log/纯图片/100ep/mambavision-T-1K minlr=3e-7/best_model_epoch_11.pth'


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    heatmap_images = []
    labels = []

    model = MambaVision(num_classes=2)
    model.load_state_dict(torch.load(model_paths))
    model.to(device)  # 确保模型在正确的设备上
    model.eval()

    # target_layer = model.model.levels[1].blocks[2].conv2
    # target_layer = model.model.levels[3].blocks[1].mixer  # 选择整个混合层
    target_layer = model.model.norm

    cam = generate_gradcam(model, image_tensor, target_layer, device)
    heatmap = overlay_heatmap(image_path, cam)
    # 创建白色背景并将热力图放置其上
    heatmap_with_background = create_white_background_image(heatmap, 224, 224)

    heatmap_images.append(heatmap_with_background)
    labels.append('mambavision')

    # output_path = '/home/zhiqinkun/project/DME/ManbaVision/log/cam'
    # save_concat_image_with_labels(heatmap_images, labels, output_path)
    output_dir = '/home/zhiqinkun/project/DME/ManbaVision/log/cam'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'output_image3.jpg')
    save_concat_image_with_labels(heatmap_images, labels, output_path)

if __name__ == "__main__":
    main()
