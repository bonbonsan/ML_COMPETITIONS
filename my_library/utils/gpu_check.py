import torch


def check_cuda():
    available = torch.cuda.is_available()
    print(f"CUDA Available: {available}")
    if available:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    check_cuda()
