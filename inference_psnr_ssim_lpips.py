import torch
from motion_diffusion import MotionDiffusion
from text_encoder import WordEncoder
from FlagEmbedding import BGEM3FlagModel
from data_process import denormalize_motion_sequence, visualize_motion_sequence, load_dataset
import os
import numpy as np
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips.LPIPS(net='vgg').to(device)

# 加载模型
motion_diffusion = MotionDiffusion().to(device)
text_encoder = WordEncoder().to(device)

# 加载预训练权重
motion_diffusion.load_state_dict(torch.load("motion_diffusion.pth", map_location=device))
text_encoder.eval()  # 文本编码器通常是冻结的

motion_diffusion.eval()


def ddim_sampling(model, text_feat, num_steps=50, eta=0.0):
    """
    DDIM采样算法，用于从扩散模型生成运动
    """
    batch_size = text_feat.shape[0]
    device = text_feat.device

    # 创建时间步序列
    timesteps = torch.linspace(model.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=device)

    # 从纯噪声开始
    x = torch.randn(batch_size, model.input_dim, device=device)

    for i, t in enumerate(timesteps):
        t_batch = t.repeat(batch_size)

        with torch.no_grad():
            # 预测噪声
            pred_noise = model(x, t_batch)

            # DDIM更新步骤
            if i < len(timesteps) - 1:
                alpha_t = model.alphas_cumprod[t]
                alpha_t_prev = model.alphas_cumprod[timesteps[i + 1]]

                # 计算x_0的预测
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
                x_0_pred = (x - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t

                # DDIM更新
                sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
                sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev)

                x = sqrt_alpha_t_prev * x_0_pred + sqrt_one_minus_alpha_t_prev * pred_noise
            else:
                # 最后一步，直接计算x_0
                alpha_t = model.alphas_cumprod[t]
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
                x = (x - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t

    return x


def ddpm_sampling(model, text_feat, num_steps=1000):
    """
    标准DDPM采样算法（更慢但可能质量更好）
    """
    batch_size = text_feat.shape[0]
    device = text_feat.device

    # 从纯噪声开始
    x = torch.randn(batch_size, model.input_dim, device=device)

    # 逐步去噪
    for t in reversed(range(0, num_steps)):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

        with torch.no_grad():
            # 预测噪声
            pred_noise = model(x, t_batch)

            # 计算去噪参数
            beta_t = model.betas[t]
            alpha_t = model.alphas_cumprod[t]
            alpha_t_prev = model.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)

            # 去噪
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

            # 计算均值
            x_0_pred = (x - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t

            # 计算后验均值
            alpha_t_prev_sqrt = torch.sqrt(alpha_t_prev)
            beta_t_sqrt = torch.sqrt(1 - alpha_t_prev)

            x_mean = alpha_t_prev_sqrt * x_0_pred + beta_t_sqrt * pred_noise

            # 添加噪声（除了最后一步）
            if t > 0:
                noise = torch.randn_like(x)
                posterior_variance = beta_t * (1. - alpha_t_prev) / (1. - alpha_t)
                x = x_mean + torch.sqrt(posterior_variance) * noise
            else:
                x = x_mean

    return x


def generate_motion_from_text(label, bge_model, use_ddim=True):
    """从文本生成运动序列"""
    # 1. 编码文本
    bge_embedding = bge_model.encode([label])['dense_vecs']
    text_tensor = torch.tensor(bge_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    text_feat = text_encoder(text_tensor)

    # 2. 选择采样算法
    with torch.no_grad():
        if use_ddim:
            generated_motion = ddim_sampling(motion_diffusion, text_feat, num_steps=50)
        else:
            generated_motion = ddpm_sampling(motion_diffusion, text_feat, num_steps=1000)

    # 3. 重塑为运动序列格式
    if generated_motion.numel() == 150 * 137 * 2:
        return generated_motion.view(150, 137, 2).detach().cpu().numpy()
    else:
        # 如果维度不匹配，需要调整
        print(f"警告：生成的运动维度为 {generated_motion.shape}")
        target_size = 150 * 137 * 2
        if generated_motion.numel() < target_size:
            # 上采样
            ratio = int(target_size / generated_motion.numel())
            generated_motion = generated_motion.repeat(1, ratio)[:, :target_size]
        else:
            # 下采样
            generated_motion = generated_motion[:, :target_size]

        return generated_motion.view(150, 137, 2).detach().cpu().numpy()


def compute_psnr_ssim_lpips(gt_path, pred_path):
    """
    计算两个文件夹下所有帧的 PSNR、SSIM、LPIPS，并返回平均值。
    """
    psnr_list, ssim_list, lpips_list = [], [], []

    if not os.path.exists(gt_path) or not os.path.exists(pred_path):
        print(f"路径不存在: {gt_path} 或 {pred_path}")
        return 0, 0, 1

    gt_files = sorted([f for f in os.listdir(gt_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    pred_files = sorted([f for f in os.listdir(pred_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if len(gt_files) == 0 or len(pred_files) == 0:
        print(f"没有找到图像文件: GT={len(gt_files)}, Pred={len(pred_files)}")
        return 0, 0, 1

    for gt_file, pred_file in zip(gt_files, pred_files):
        try:
            gt_img = np.array(Image.open(os.path.join(gt_path, gt_file)).convert("L"))
            pred_img = np.array(Image.open(os.path.join(pred_path, pred_file)).convert("L"))

            # 确保图像大小相同
            if gt_img.shape != pred_img.shape:
                # 调整到相同大小
                min_height = min(gt_img.shape[0], pred_img.shape[0])
                min_width = min(gt_img.shape[1], pred_img.shape[1])
                gt_img = gt_img[:min_height, :min_width]
                pred_img = pred_img[:min_height, :min_width]

            # 计算 PSNR
            psnr_value = psnr(gt_img, pred_img, data_range=255)
            if not np.isinf(psnr_value) and not np.isnan(psnr_value):
                psnr_list.append(psnr_value)

            # 计算 SSIM
            ssim_value = ssim(gt_img, pred_img, data_range=255)
            if not np.isinf(ssim_value) and not np.isnan(ssim_value):
                ssim_list.append(ssim_value)

            # 计算 LPIPS
            # 转换为RGB（LPIPS需要3通道）
            gt_tensor = torch.from_numpy(gt_img).float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0) / 255.0
            pred_tensor = torch.from_numpy(pred_img).float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0) / 255.0

            gt_tensor, pred_tensor = gt_tensor.to(device), pred_tensor.to(device)
            lpips_value = lpips_model(gt_tensor, pred_tensor).item()
            if not np.isinf(lpips_value) and not np.isnan(lpips_value):
                lpips_list.append(lpips_value)

        except Exception as e:
            print(f"处理图像 {gt_file}/{pred_file} 时出错: {e}")
            continue

    # 计算平均值，处理空列表的情况
    avg_psnr = np.mean(psnr_list) if psnr_list else 0
    avg_ssim = np.mean(ssim_list) if ssim_list else 0
    avg_lpips = np.mean(lpips_list) if lpips_list else 1

    return avg_psnr, avg_ssim, avg_lpips


if __name__ == "__main__":
    # 初始化BGE模型
    bge_model = BGEM3FlagModel('/data/SIGNVAE-main/models/bge-m3', use_fp16=True)

    # 加载数据
    data_path = "data"
    motions, labels = load_dataset(data_path)
    print(f"数据加载完成，共 {len(motions)} 个样本")

    output_base_path = "./output_diffusion_psnr_ssim_lpips"
    total_psnr, total_ssim, total_lpips = 0, 0, 0
    results = []

    # 确保输出目录存在
    os.makedirs(output_base_path, exist_ok=True)

    print("开始生成运动序列和评估图像质量...")
    for i, label in tqdm(enumerate(labels), total=len(labels), desc="生成和评估"):
        # 生成预测结果
        motion_sequence = generate_motion_from_text(label, bge_model, use_ddim=True)
        motion_sequence_re = denormalize_motion_sequence(motion_sequence)

        # GT 关键点
        motion_sequence_gt = denormalize_motion_sequence(motions[i])

        # 可视化并保存图像
        try:
            visualize_motion_sequence(motion_sequence_gt, output_base_path, i, "gt")
            visualize_motion_sequence(motion_sequence_re, output_base_path, i, "pred")
        except Exception as e:
            print(f"可视化样本 {i} 时出错: {e}")
            continue

        # 计算评估指标
        gt_path = os.path.join(output_base_path, f"sample_{i:03d}", "gt")
        pred_path = os.path.join(output_base_path, f"sample_{i:03d}", "pred")

        psnr_avg, ssim_avg, lpips_avg = compute_psnr_ssim_lpips(gt_path, pred_path)

        total_psnr += psnr_avg
        total_ssim += ssim_avg
        total_lpips += lpips_avg

        results.append(
            f"Sample {i + 1} ({label}): PSNR = {psnr_avg:.6f}, SSIM = {ssim_avg:.6f}, LPIPS = {lpips_avg:.6f}")

    # 计算数据集的平均指标
    if len(labels) > 0:
        avg_psnr = total_psnr / len(labels)
        avg_ssim = total_ssim / len(labels)
        avg_lpips = total_lpips / len(labels)

        results.append(f"\nAverage PSNR: {avg_psnr:.6f}")
        results.append(f"Average SSIM: {avg_ssim:.6f}")
        results.append(f"Average LPIPS: {avg_lpips:.6f}")

        # 保存结果
        with open("diffusion_image_quality_results.txt", "w") as f:
            f.write("\n".join(results) + "\n")

        print(f"\n计算完成！")
        print(f"平均 PSNR: {avg_psnr:.6f}")
        print(f"平均 SSIM: {avg_ssim:.6f}")
        print(f"平均 LPIPS: {avg_lpips:.6f}")
        print("结果已保存到 diffusion_image_quality_results.txt")