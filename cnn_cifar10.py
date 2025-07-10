import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# 定义卷积神经网络（以ResNet18为基础，适合CIFAR-10）
from torchvision.models import resnet18
from torchvision.datasets.utils import download_url

class CIFAR10_Mirror(torchvision.datasets.CIFAR10):
    def _download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        # 国内镜像地址
        url = 'https://mirrors.tuna.tsinghua.edu.cn/gitlab.com/ultralytics/yolov5/-/raw/master/data/cifar-10-python.tar.gz'
        download_url(url, self.root, self.filename, self.tgz_md5)

def get_resnet18_for_cifar10(num_classes=10):
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# 训练函数
def train(model, device, trainloader, testloader, epochs=50, lr=0.1, save_path='cifar10_resnet18.pth'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    # 动态学习率策略：CosineAnnealingWarmRestarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # 第一次重启的周期
        T_mult=2,  # 每次重启后周期翻倍
        eta_min=1e-6  # 最小学习率
    )
    
    # 早停机制
    patience = 15
    best_acc = 0.0
    patience_counter = 0
    
    print("开始训练...")
    print(f"初始学习率: {lr}")
    print(f"训练轮数: {epochs}")
    print(f"早停耐心值: {patience}")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 计算训练准确率
        train_acc = 100 * correct / total
        avg_loss = running_loss / len(trainloader)
        
        # 测试准确率
        test_acc = test(model, device, testloader)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, LR: {current_lr:.6f}')
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            print(f'保存最佳模型，测试准确率: {best_acc:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'早停触发！{patience} 轮未改善，停止训练')
                break
    
    print(f'训练完成！最佳测试准确率: {best_acc:.2f}%')
    return best_acc

def test(model, device, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 推理函数
def predict_image(model, device, image_path, class_names):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CIFAR10分类训练与推理')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--predict', type=str, help='对指定图片进行推理')
    parser.add_argument('--model', type=str, default='cifar10_resnet18.pth', help='模型权重文件路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.1, help='初始学习率')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']

    if args.train:
        # 增强的训练数据变换
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ])
        
        # 测试数据变换（无增强）
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = CIFAR10_Mirror(root='./data', train=True, download=False, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = CIFAR10_Mirror(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
        model = get_resnet18_for_cifar10().to(device)
        
        # 训练参数
        best_acc = train(model, device, trainloader, testloader, epochs=args.epochs, lr=args.lr)
        print(f"最终最佳准确率: {best_acc:.2f}%")
    elif args.predict:
        model = get_resnet18_for_cifar10().to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        
        # 检查是文件还是目录
        if os.path.isfile(args.predict):
            # 单个文件推理
            pred = predict_image(model, device, args.predict, class_names)
            print(f'图片 {args.predict} 的预测类别为: {pred}')
        elif os.path.isdir(args.predict):
            # 目录批量推理
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            for filename in os.listdir(args.predict):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    image_path = os.path.join(args.predict, filename)
                    try:
                        pred = predict_image(model, device, image_path, class_names)
                        print(f'图片 {filename} 的预测类别为: {pred}')
                    except Exception as e:
                        print(f'处理图片 {filename} 时出错: {e}')
        else:
            print(f'错误: {args.predict} 不是有效的文件或目录')
    else:
        print('请指定 --train 进行训练，或 --predict <图片路径> 进行推理。') 