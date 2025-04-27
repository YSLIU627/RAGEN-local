import ray
import time

# 连接到已有的Ray集群
ray.init(address="auto")

@ray.remote
def test_func(x):
    time.sleep(1)  # 模拟一点小延迟
    return x * x

# 提交10个任务
futures = [test_func.remote(i) for i in range(10)]

# 等待所有任务完成
results = ray.get(futures)

print("Test finished! Results:", results)

# 正常关闭Ray
ray.shutdown()
