import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def concat_images_vertically(image_dir='./predict', output_path='concatenated_images.png', 
                           target_width=300, spacing=10, background_color=(255, 255, 255)):
    """
    将目录下的所有图片拼接成一竖排
    
    参数:
    image_dir: 图片目录路径
    output_path: 输出文件路径
    target_width: 目标宽度（像素）
    spacing: 图片间距（像素）
    background_color: 背景颜色 (R, G, B)
    """
    
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    # 获取所有图片文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if not image_files:
        print(f"在 {image_dir} 目录下未找到图片文件")
        return None
    
    # 按文件名排序
    image_files.sort()
    print(f"找到 {len(image_files)} 张图片:")
    for f in image_files:
        print(f"  - {os.path.basename(f)}")
    
    # 加载并调整所有图片
    images = []
    total_height = 0
    
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')
            
            # 计算缩放比例，保持宽高比
            width, height = img.size
            scale = target_width / width
            new_height = int(height * scale)
            
            # 调整图片大小
            img_resized = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
            images.append(img_resized)
            total_height += new_height
            
            print(f"调整图片 {os.path.basename(img_path)}: {width}x{height} -> {target_width}x{new_height}")
            
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")
            continue
    
    if not images:
        print("没有成功加载任何图片")
        return None
    
    # 计算总高度（包括间距）
    total_height += spacing * (len(images) - 1)
    
    # 创建拼接后的图片
    concatenated_img = Image.new('RGB', (target_width, total_height), background_color)
    
    # 拼接图片
    current_y = 0
    for i, img in enumerate(images):
        concatenated_img.paste(img, (0, current_y))
        current_y += img.height + spacing
    
    # 保存拼接后的图片
    concatenated_img.save(output_path)
    print(f"\n拼接完成！输出文件: {output_path}")
    print(f"最终尺寸: {target_width}x{total_height}")
    
    return concatenated_img

def display_concatenated_images(image_path, figsize=(12, 20)):
    """
    显示拼接后的图片
    """
    try:
        img = Image.open(image_path)
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.axis('off')
        plt.title('拼接后的图片', fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"显示图片时出错: {e}")

def create_image_grid_with_labels(image_dir='./predict', output_path='image_grid_with_labels.png',
                                 images_per_row=3, target_size=(200, 200), spacing=20):
    """
    创建带标签的图片网格
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if not image_files:
        print(f"在 {image_dir} 目录下未找到图片文件")
        return None
    
    image_files.sort()
    
    # 计算网格尺寸
    num_images = len(image_files)
    num_rows = (num_images + images_per_row - 1) // images_per_row
    
    # 计算总尺寸
    total_width = images_per_row * target_size[0] + (images_per_row - 1) * spacing
    total_height = num_rows * target_size[1] + (num_rows - 1) * spacing + 50  # 额外空间用于标签
    
    # 创建画布
    fig, axes = plt.subplots(num_rows, images_per_row, figsize=(15, 5 * num_rows))
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    # 填充图片
    for i, img_path in enumerate(image_files):
        row = i // images_per_row
        col = i % images_per_row
        
        try:
            img = Image.open(img_path).convert('RGB')
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            
            axes[row, col].imshow(img_resized)
            axes[row, col].set_title(os.path.basename(img_path), fontsize=10)
            axes[row, col].axis('off')
            
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")
            axes[row, col].text(0.5, 0.5, 'Error', ha='center', va='center')
            axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_images, num_rows * images_per_row):
        row = i // images_per_row
        col = i % images_per_row
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"网格图片已保存: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='图片拼接工具')
    parser.add_argument('--input_dir', type=str, default='./predict', help='输入图片目录')
    parser.add_argument('--output', type=str, default='concatenated_images.png', help='输出文件路径')
    parser.add_argument('--width', type=int, default=300, help='目标宽度')
    parser.add_argument('--spacing', type=int, default=10, help='图片间距')
    parser.add_argument('--mode', type=str, choices=['vertical', 'grid'], default='vertical', 
                       help='拼接模式: vertical(竖排) 或 grid(网格)')
    parser.add_argument('--display', action='store_true', help='显示拼接后的图片')
    
    args = parser.parse_args()
    
    if args.mode == 'vertical':
        # 竖排拼接
        result = concat_images_vertically(
            image_dir=args.input_dir,
            output_path=args.output,
            target_width=args.width,
            spacing=args.spacing
        )
        
        if args.display and result:
            display_concatenated_images(args.output)
    
    elif args.mode == 'grid':
        # 网格拼接
        create_image_grid_with_labels(
            image_dir=args.input_dir,
            output_path=args.output
        ) 