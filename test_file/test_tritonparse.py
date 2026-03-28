import torch
import triton
import triton.language as tl
import tritonparse.structured_logging
import tritonparse.parse.utils

# Initialize logging (see Standard Setup Pattern above)
log_path = "./logs/"
tritonparse.structured_logging.init(log_path, enable_trace_launch=True)

@triton.jit
def triton_add(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    """NPU版本的向量加法内核"""
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB

    for loop1 in range(loops1):
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr0 + (x0), tmp2, None)

def tensor_add_npu(a, b, ncore=2, xblock=1024, xblock_sub=1024):
    """NPU版本的张量加法包装函数"""
    c = torch.empty_like(a, device="npu")
    # 调用NPU版本的内核
    triton_add[ncore, 1, 1](a, b, c, xblock, xblock_sub)
    return c

# Example usage
if __name__ == "__main__":
    print("🚀 开始测试NPU版本的Triton向量加法内核")

    # 设置测试参数 (参考用户提供的参数)
    dtype = torch.float32
    shape = (2, 4096, 8)
    ncore = 2
    xblock = 32768
    xblock_sub = 1024
    # 创建测试数据
    a = torch.randn(shape, dtype=dtype, device="npu")
    b = torch.randn(shape, dtype=dtype, device="npu")
    c_npu = tensor_add_npu(a, b, ncore, xblock, xblock_sub)

    print("🎉 NPU内核执行完成！")
    print("📁 日志已保存到 ./logs/ 目录")

    # Parse the generated logs (see Standard Setup Pattern above)
    tritonparse.parse.utils.unified_parse(source=log_path, out="./parsed_output", overwrite=True)