import argparse
import time
import torch


def allocate_vram(target_ratio: float, dtype=torch.float16):
    """
    按目标比例占用显存。
    target_ratio: 显存占用比例，例如 0.90 表示占用约 90% 可用显存。
    """
    device = torch.device("cuda:0")
    reserved_tensors = []

    free_mem, total_mem = torch.cuda.mem_get_info(0)
    target_bytes = int(total_mem * target_ratio)

    print(f"GPU 总显存: {total_mem / 1024**3:.2f} GB")
    print(f"当前可用显存: {free_mem / 1024**3:.2f} GB")
    print(f"目标占用显存: {target_bytes / 1024**3:.2f} GB")

    # 每块 256MB，逐步申请，避免一次性 OOM
    chunk_bytes = 256 * 1024 * 1024
    element_size = torch.tensor([], dtype=dtype).element_size()
    chunk_elements = chunk_bytes // element_size

    allocated = torch.cuda.memory_allocated(device)

    while allocated < target_bytes:
        try:
            tensor = torch.empty(
                chunk_elements,
                dtype=dtype,
                device=device
            )
            tensor.fill_(1)
            reserved_tensors.append(tensor)

            allocated = torch.cuda.memory_allocated(device)
            print(f"已占用显存: {allocated / 1024**3:.2f} GB", end="\r")

        except torch.cuda.OutOfMemoryError:
            print("\n显存已接近上限，停止继续申请。")
            torch.cuda.empty_cache()
            break

    print("\n显存占用完成。")
    return reserved_tensors


def stress_compute(matrix_size: int, duration: int):
    """
    通过矩阵乘法持续拉高 GPU 计算利用率。
    matrix_size 越大，GPU 计算压力越高。
    duration: 持续时间，单位秒。
    """
    device = torch.device("cuda:0")

    a = torch.randn(
        matrix_size,
        matrix_size,
        device=device,
        dtype=torch.float16
    )
    b = torch.randn(
        matrix_size,
        matrix_size,
        device=device,
        dtype=torch.float16
    )

    print(f"开始 GPU 计算压力测试，矩阵大小: {matrix_size} x {matrix_size}")
    print(f"持续时间: {duration} 秒")

    start_time = time.time()
    count = 0

    while time.time() - start_time < duration:
        c = torch.matmul(a, b)
        a = c
        count += 1

        if count % 20 == 0:
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            print(f" {elapsed:.1f} 秒，{count}")

    torch.cuda.synchronize()
    print("GPU 计算压力测试结束。")


def main():
    parser = argparse.ArgumentParser(description="GPU 显存与算力压力测试脚本")

    parser.add_argument(
        "--vram-ratio",
        type=float,
        default=0.35,
        help="目标显存占用比例，默认 0.90"
    )

    parser.add_argument(
        "--matrix-size",
        type=int,
        default=4192,
        help="矩阵大小，默认 8192"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=999999999999,
        help="持续时间，单位秒，默认 300 秒"
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA GPU，请确认已正确安装 NVIDIA 驱动和 PyTorch CUDA 版本。")

    if args.vram_ratio <= 0 or args.vram_ratio >= 1:
        raise ValueError("vram-ratio 建议设置在 0 到 1 之间，例如 0.90。")

    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")

    reserved_tensors = allocate_vram(args.vram_ratio)
    stress_compute(args.matrix_size, args.duration)

    # 保留变量，防止显存提前释放
    _ = reserved_tensors


if __name__ == "__main__":
    main()