import torch
from motion_diffusion import MotionDiffusion
from text_encoder import WordEncoder
from FlagEmbedding import BGEM3FlagModel
from data_process import load_dataset
import os
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
motion_diffusion = MotionDiffusion().to(device)
text_encoder = WordEncoder().to(device)

# 加载预训练权重
motion_diffusion.load_state_dict(torch.load("motion_diffusion.pth", map_location=device))
text_encoder.eval()  # 文本编码器通常是冻结的

motion_diffusion.eval()

# 检查并修复噪声调度器
if motion_diffusion.alphas_cumprod.max() == 0:
    print("警告：alphas_cumprod初始化有问题，重新初始化噪声调度器")

    # 重新初始化正确的余弦调度器
    steps = motion_diffusion.num_timesteps

    # 使用标准的DDPM调度器
    betas = torch.linspace(0.0001, 0.02, steps, device=device)

    # 或者使用修复的余弦调度器
    # s = 0.008  # 小常数
    # t = torch.linspace(0, 1, steps + 1, device=device)
    # ft = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
    # alphas_cumprod = ft / ft[0]
    # betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    # betas = torch.clamp(betas, 0, 0.999)

    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas = torch.sqrt(alphas)

    # 更新模型的调度器参数
    motion_diffusion.register_buffer('betas', betas)
    motion_diffusion.register_buffer('alphas_cumprod', alphas_cumprod)
    motion_diffusion.register_buffer('sqrt_alphas', sqrt_alphas)

    print(f"修复后的alphas_cumprod范围: [{alphas_cumprod.min():.6f}, {alphas_cumprod.max():.6f}]")


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

    # 添加数值稳定性检查
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("警告：初始噪声包含NaN或Inf值")
        x = torch.zeros_like(x)
        x.normal_(0, 0.1)  # 使用更小的标准差

    for i, t in enumerate(timesteps):
        t_batch = t.repeat(batch_size)

        with torch.no_grad():
            # 预测噪声，根据原始代码只传递x和t
            pred_noise = model(x, t_batch)

            # 检查预测的噪声是否包含NaN
            if torch.isnan(pred_noise).any() or torch.isinf(pred_noise).any():
                print(f"警告：在时间步 {t} 预测的噪声包含NaN或Inf值")
                pred_noise = torch.nan_to_num(pred_noise, nan=0.0, posinf=0.0, neginf=0.0)

            # DDIM更新步骤
            if i < len(timesteps) - 1:
                alpha_t = model.alphas_cumprod[t]
                alpha_t_prev = model.alphas_cumprod[timesteps[i + 1]]

                # 添加数值稳定性检查
                if alpha_t <= 0 or alpha_t >= 1:
                    print(f"警告：alpha_t值异常: {alpha_t}")
                    continue

                # 计算x_0的预测
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

                # 避免除零
                if sqrt_alpha_t < 1e-8:
                    sqrt_alpha_t = torch.tensor(1e-8, device=device)

                x_0_pred = (x - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t

                # DDIM更新
                sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
                sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev)

                x = sqrt_alpha_t_prev * x_0_pred + sqrt_one_minus_alpha_t_prev * pred_noise

                # 检查更新后的x
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"警告：在时间步 {t} 更新后的x包含NaN或Inf值")
                    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                # 最后一步，直接计算x_0
                alpha_t = model.alphas_cumprod[t]

                if alpha_t <= 0 or alpha_t >= 1:
                    print(f"警告：最后一步alpha_t值异常: {alpha_t}")
                    break

                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

                # 避免除零
                if sqrt_alpha_t < 1e-8:
                    sqrt_alpha_t = torch.tensor(1e-8, device=device)

                x = (x - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t

    # 最终检查
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("警告：最终生成的运动包含NaN或Inf值，尝试修复")
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    return x


def generate_motion_from_text(label, bge_model):
    """从文本生成运动序列（当前版本不使用文本条件）"""
    # 注意：当前模型实际上不使用文本条件，只是生成随机运动
    print(f"注意：当前模型不使用文本条件'{label}'，生成随机运动")

    # 创建一个虚拟的文本特征，只是为了保持接口一致
    dummy_text_feat = torch.zeros(1, 256).to(device)  # 假设文本特征维度为256

    # 2. 使用DDIM采样生成运动
    with torch.no_grad():
        generated_motion = ddim_sampling(motion_diffusion, dummy_text_feat, num_steps=50)

    # 3. 重塑为运动序列格式
    batch_size = generated_motion.shape[0]

    # 检查生成的运动维度
    if generated_motion.numel() == batch_size * 41100:
        # 移除batch维度
        if batch_size == 1:
            generated_motion = generated_motion.squeeze(0)

        print(f"生成运动原始形状: {generated_motion.shape}")

        # 41100应该对应150*137*2 = 41100
        target_size = 150 * 137 * 2  # 41100

        # 确保数据是有效的
        if torch.isnan(generated_motion).any() or torch.isinf(generated_motion).any():
            print("警告：生成的运动包含NaN或Inf值，使用零填充")
            generated_motion = torch.zeros_like(generated_motion)

        # 转换为numpy并重塑
        motion_np = generated_motion.detach().cpu().numpy()

        # 重塑为(150, 137, 2)
        if motion_np.size == target_size:
            return motion_np.reshape(150, 137, 2)
        else:
            print(f"警告：期望大小{target_size}，实际大小{motion_np.size}")
            # 如果大小不匹配，创建零矩阵
            return np.zeros((150, 137, 2))
    else:
        print(f"错误：无法处理的运动维度 {generated_motion.shape}")
        return np.zeros((150, 137, 2))


def safe_mse(y_true, y_pred):
    """安全版本的MSE，处理NaN和Inf值"""
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        print("警告：输入包含NaN值")
        return np.nan

    diff = y_true - y_pred
    if np.any(np.isnan(diff)) or np.any(np.isinf(diff)):
        print("警告：检测到NaN或Inf值")
        return np.nan

    return np.mean(diff ** 2)


def safe_mae(y_true, y_pred):
    """安全版本的MAE，处理NaN和Inf值"""
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        print("警告：输入包含NaN值")
        return np.nan

    diff = np.abs(y_true - y_pred)
    if np.any(np.isnan(diff)) or np.any(np.isinf(diff)):
        print("警告：检测到NaN或Inf值")
        return np.nan

    return np.mean(diff)


def safe_nmse(y_true, y_pred):
    """安全版本的NMSE，避免除零错误"""
    # 检查输入是否有效
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        print("警告：输入包含NaN值")
        return np.nan

    # 计算数据范围
    y_true_flat = y_true.flatten()
    data_range = np.max(y_true_flat) - np.min(y_true_flat)

    # 避免除零错误
    if data_range == 0 or np.isclose(data_range, 0, atol=1e-8):
        print("警告：数据范围接近0，返回0")
        return 0.0

    mse_value = safe_mse(y_true, y_pred)
    if np.isnan(mse_value):
        return np.nan

    return mse_value / (data_range ** 2)


def safe_nmae(y_true, y_pred):
    """安全版本的NMAE，避免除零错误"""
    # 检查输入是否有效
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        print("警告：输入包含NaN值")
        return np.nan

    # 计算数据范围
    y_true_flat = y_true.flatten()
    data_range = np.max(y_true_flat) - np.min(y_true_flat)

    # 避免除零错误
    if data_range == 0 or np.isclose(data_range, 0, atol=1e-8):
        print("警告：数据范围接近0，返回0")
        return 0.0

    mae_value = safe_mae(y_true, y_pred)
    if np.isnan(mae_value):
        return np.nan

    return mae_value / data_range


def frechet_distance(mu1, sigma1, mu2, sigma2):
    """计算Fréchet距离"""
    try:
        diff = mu1 - mu2
        # 添加小值避免数值不稳定
        epsilon = 1e-6

        # 确保协方差矩阵是正定的
        sigma1 = sigma1 + epsilon * np.eye(sigma1.shape[0])
        sigma2 = sigma2 + epsilon * np.eye(sigma2.shape[0])

        # 使用更稳定的方式计算协方差矩阵的平方根
        from scipy.linalg import sqrtm
        covmean = sqrtm(sigma1.dot(sigma2))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    except Exception as e:
        print(f"计算Fréchet Distance时出错: {e}")
        return np.nan


def compute_motion_statistics(motions):
    """计算运动序列的统计信息"""
    try:
        motions_flat = motions.reshape(len(motions), -1)
        mu = np.mean(motions_flat, axis=0)
        # 添加正则化项避免奇异矩阵
        sigma = np.cov(motions_flat, rowvar=False)

        # 确保协方差矩阵是数值稳定的
        min_eig = np.min(np.real(np.linalg.eigvals(sigma)))
        if min_eig < 0:
            sigma -= 1.1 * min_eig * np.eye(sigma.shape[0])

        return mu, sigma
    except Exception as e:
        print(f"计算统计信息时出错: {e}")
        return None, None


# 主程序
if __name__ == "__main__":
    # 初始化BGE模型
    bge_model = BGEM3FlagModel('/data/SIGNVAE-main/models/bge-m3', use_fp16=True)

    # 加载数据
    data_path = "data"
    motions, labels = load_dataset(data_path)
    print(f"数据加载完成，共 {len(motions)} 个样本")

    # 检查数据样本
    print(f"真实运动数据形状示例: {motions[0].shape}")
    print(f"数据范围: [{np.min(motions[0]):.6f}, {np.max(motions[0]):.6f}]")
    print(f"标签示例: {labels[0]}")

    # 检查模型参数
    print(f"模型input_dim: {motion_diffusion.input_dim}")
    print(f"模型num_timesteps: {motion_diffusion.num_timesteps}")

    # 检查alphas_cumprod是否正确初始化
    if hasattr(motion_diffusion, 'alphas_cumprod'):
        print(
            f"alphas_cumprod范围: [{motion_diffusion.alphas_cumprod.min():.6f}, {motion_diffusion.alphas_cumprod.max():.6f}]")
        print(f"alphas_cumprod设备: {motion_diffusion.alphas_cumprod.device}")
    else:
        print("警告：模型没有alphas_cumprod属性")

    # 测试一个简单的前向传播
    print("\n测试模型前向传播...")
    test_input = torch.randn(1, motion_diffusion.input_dim).to(device)
    test_t = torch.randint(0, motion_diffusion.num_timesteps, (1,)).to(device)

    with torch.no_grad():
        try:
            test_output = motion_diffusion(test_input, test_t)
            print(f"测试输出形状: {test_output.shape}")
            print(f"测试输出范围: [{test_output.min():.6f}, {test_output.max():.6f}]")
            if torch.isnan(test_output).any():
                print("警告：测试输出包含NaN")
            if torch.isinf(test_output).any():
                print("警告：测试输出包含Inf")
        except Exception as e:
            print(f"模型前向传播测试失败: {e}")
            import traceback

            traceback.print_exc()

    total_nmse, total_nmae = 0, 0
    total_mse, total_mae = 0, 0
    results = []
    valid_sample_count = 0  # 跟踪有效样本数量

    # 分别存储真实和生成的运动
    real_motions = []
    generated_motions = []

    print("开始生成运动序列...")
    for i, label in enumerate(labels[:100]):  # 先测试前100个样本
        try:
            print(f"正在处理样本 {i}: {label}")

            # 生成运动序列
            motion_sequence = generate_motion_from_text(label, bge_model)

            # 检查生成的运动是否有效
            if motion_sequence is None:
                print(f"样本 {i} 生成失败，跳过")
                continue

            # 检查生成的运动是否包含NaN
            if np.any(np.isnan(motion_sequence)):
                print(f"警告：样本 {i} 生成的运动包含NaN值，尝试修复")
                motion_sequence = np.nan_to_num(motion_sequence, nan=0.0, posinf=0.0, neginf=0.0)
                # 如果修复后仍然全部是0或异常，跳过
                if np.all(motion_sequence == 0) or np.any(np.isnan(motion_sequence)):
                    print(f"样本 {i} 修复失败，跳过")
                    continue

            generated_motions.append(motion_sequence)

            # 计算与真实运动的差异
            real_motion = motions[i]
            real_motions.append(real_motion)

            # 确保形状匹配
            if real_motion.shape != motion_sequence.shape:
                print(f"警告：样本 {i} 的运动形状不匹配")
                print(f"真实运动: {real_motion.shape}, 生成运动: {motion_sequence.shape}")
                continue

            # 检查真实运动数据有效性
            if np.any(np.isnan(real_motion)):
                print(f"警告：样本 {i} 的真实运动包含NaN值，跳过")
                continue

            # 计算指标
            mse_value = safe_mse(real_motion, motion_sequence)
            mae_value = safe_mae(real_motion, motion_sequence)
            nmse_value = safe_nmse(real_motion, motion_sequence)
            nmae_value = safe_nmae(real_motion, motion_sequence)

            # 检查计算结果是否有效
            if not (np.isnan(mse_value) or np.isnan(mae_value) or np.isnan(nmse_value) or np.isnan(nmae_value)):
                total_mse += mse_value
                total_mae += mae_value
                total_nmse += nmse_value
                total_nmae += nmae_value
                valid_sample_count += 1

                results.append(
                    f"Sample {i + 1} ({label}): MSE = {mse_value:.6f}, MAE = {mae_value:.6f}, NMSE = {nmse_value:.6f}, NMAE = {nmae_value:.6f}")
                print(f"样本 {i} 成功处理: MSE={mse_value:.6f}, MAE={mae_value:.6f}")
            else:
                print(f"警告：样本 {i} 的计算结果包含NaN，跳过")

            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1}/100 个样本，有效样本: {valid_sample_count}")

        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
            import traceback

            traceback.print_exc()
            continue

    # 计算平均指标
    if valid_sample_count > 0:
        avg_mse = total_mse / valid_sample_count
        avg_mae = total_mae / valid_sample_count
        avg_nmse = total_nmse / valid_sample_count
        avg_nmae = total_nmae / valid_sample_count

        print(f"\n评估结果（基于{valid_sample_count}个有效样本）:")
        print(f"平均 MSE: {avg_mse:.6f}")
        print(f"平均 MAE: {avg_mae:.6f}")
        print(f"平均 NMSE: {avg_nmse:.6f}")
        print(f"平均 NMAE: {avg_nmae:.6f}")

        # 计算Fréchet Inception Distance (FID) 类似的度量
        if len(real_motions) > 1 and len(generated_motions) > 1:
            try:
                real_mu, real_sigma = compute_motion_statistics(np.array(real_motions))
                gen_mu, gen_sigma = compute_motion_statistics(np.array(generated_motions))

                if real_mu is not None and real_sigma is not None and gen_mu is not None and gen_sigma is not None:
                    fd = frechet_distance(real_mu, real_sigma, gen_mu, gen_sigma)
                    if not np.isnan(fd):
                        results.append(f"\nFréchet Distance: {fd:.6f}")
                        print(f"Fréchet Distance: {fd:.6f}")
                    else:
                        results.append(f"\nFréchet Distance: 计算失败")
                        print("Fréchet Distance: 计算失败")
                else:
                    results.append(f"\nFréchet Distance: 统计信息计算失败")
                    print("Fréchet Distance: 统计信息计算失败")
            except Exception as e:
                results.append(f"\nFréchet Distance: 计算时出错 - {e}")
                print(f"计算Fréchet Distance时出错: {e}")

        results.append(f"\n有效样本数: {valid_sample_count}/100")
        results.append(f"Average MSE: {avg_mse:.6f}")
        results.append(f"Average MAE: {avg_mae:.6f}")
        results.append(f"Average NMSE: {avg_nmse:.6f}")
        results.append(f"Average NMAE: {avg_nmae:.6f}")

        # 保存结果
        with open("diffusion_evaluation_results_nmse_nmae.txt", "w") as f:
            f.write("\n".join(results) + "\n")

        print("\n结果已保存到 diffusion_evaluation_results_nmse_nmae.txt")
    else:
        print("错误：没有有效的样本进行评估")
        with open("diffusion_evaluation_results_nmse_nmae.txt", "w") as f:
            f.write("错误：没有有效的样本进行评估\n")

    # 清理资源
    try:
        print("正在清理资源...")

        # 1. 显式删除BGE模型
        if 'bge_model' in locals():
            del bge_model

        # 2. 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # 3. 强制垃圾回收
        import gc

        gc.collect()

        # 4. 如果还有问题，可以尝试关闭所有多进程
        # 这是一个更激进的方法，通常不需要
        # import multiprocessing
        # multiprocessing.set_start_method('spawn', force=True)

        print("资源清理完成")

    except Exception as e:
        print(f"清理资源时出错: {e}")

    finally:
        # 5. 确保程序正常退出
        print("推理完成，程序即将退出...")
        import sys

        sys.exit(0)