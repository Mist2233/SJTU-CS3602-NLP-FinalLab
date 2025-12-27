#!/usr/bin/env python3
"""
数据集下载脚本
下载 WikiText-2 和 PG-19 样本数据集到本地
"""

import os
import sys
from datasets import load_dataset
import datasets


def setup_environment():
    """配置 HuggingFace 环境"""
    # 使用镜像加速（中国大陆用户）
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # 设置缓存目录
    current_dir = os.getcwd()
    cache_dir = os.path.join(current_dir, "hf_cache")
    os.environ["HF_HOME"] = cache_dir

    # 创建缓存目录
    os.makedirs(cache_dir, exist_ok=True)
    datasets_dir = os.path.join(cache_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    print(f"📁 缓存目录: {cache_dir}")
    print(f"🌐 使用镜像: {os.environ['HF_ENDPOINT']}")
    print("-" * 60)

    return datasets_dir


def download_wikitext(datasets_dir):
    """下载 WikiText-2 数据集"""
    print("\n📥 下载 WikiText-2 数据集...")
    print("   用途: PPL (困惑度) 评估")

    try:
        # 下载数据集
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=datasets_dir)

        # 验证下载
        test_size = len(dataset["test"])
        train_size = len(dataset["train"])

        print(f"   ✅ 下载成功！")
        print(f"      - Train split: {train_size} 样本")
        print(f"      - Test split: {test_size} 样本")
        print(f"      - 保存位置: {datasets_dir}/wikitext/")

        return True
    except Exception as e:
        print(f"   ❌ 下载失败: {str(e)}")
        return False


def download_pg19_sample(datasets_dir):
    """下载 PG-19 样本数据"""
    print("\n📥 下载 PG-19 数据集样本...")
    print("   用途: 长文本生成速度测试")

    try:
        # 创建 pg19_sample 目录
        pg19_dir = os.path.join(datasets_dir, "pg19_sample")
        os.makedirs(pg19_dir, exist_ok=True)

        # 使用流式加载获取一个样本
        print("   正在从 PG-19 数据集获取样本...")
        dataset = load_dataset(
            "pg19", split="train", streaming=True, trust_remote_code=True
        )

        # 获取第一个样本
        sample = next(iter(dataset))
        book_text = sample["text"]

        # 保存样本到本地
        sample_file = os.path.join(pg19_dir, "pg19_sample.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(book_text)

        # 验证
        file_size = os.path.getsize(sample_file)
        print(f"   ✅ 下载成功！")
        print(f"      - 样本大小: {file_size / 1024:.1f} KB")
        print(f"      - 文本长度: {len(book_text):,} 字符")
        print(f"      - 保存位置: {pg19_dir}/pg19_sample.txt")

        return True
    except Exception as e:
        print(f"   ❌ 下载失败: {str(e)}")
        print(f"   💡 提示: PG-19 需要信任远程代码，已自动设置")
        return False


def verify_datasets(datasets_dir):
    """验证数据集完整性"""
    print("\n🔍 验证数据集完整性...")

    # 检查 WikiText
    wikitext_path = os.path.join(datasets_dir, "wikitext")
    wikitext_ok = os.path.exists(wikitext_path) and len(os.listdir(wikitext_path)) > 0

    # 检查 PG-19
    pg19_path = os.path.join(datasets_dir, "pg19_sample", "pg19_sample.txt")
    pg19_ok = os.path.exists(pg19_path) and os.path.getsize(pg19_path) > 0

    print(f"   WikiText-2: {'✅ 完整' if wikitext_ok else '❌ 缺失'}")
    print(f"   PG-19 样本: {'✅ 完整' if pg19_ok else '❌ 缺失'}")

    return wikitext_ok and pg19_ok


def main():
    """主函数"""
    print("=" * 60)
    print(" 数据集下载工具 ".center(60, "="))
    print("=" * 60)

    # 1. 配置环境
    datasets_dir = setup_environment()

    # 2. 下载 WikiText-2
    wikitext_success = download_wikitext(datasets_dir)

    # 3. 下载 PG-19 样本
    pg19_success = download_pg19_sample(datasets_dir)

    # 4. 验证完整性
    all_ok = verify_datasets(datasets_dir)

    # 5. 输出总结
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ 所有数据集下载完成！")
        print("\n下一步:")
        print("  1. 运行 python benchmark_streaming.py 进行测试")
        print("  2. 或运行 python run_pythia.py 快速验证模型")
    else:
        print("⚠️ 部分数据集下载失败")
        print("\n请检查:")
        print("  1. 网络连接是否正常")
        print("  2. 是否设置了正确的镜像地址")
        print("  3. 磁盘空间是否充足")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
