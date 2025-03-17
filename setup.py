from setuptools import setup, find_packages

setup(
    name="AIROOF",  # 你的项目名称
    version="0.1.0",  # 版本号
    author="SICHENG YANG, ZIHAO LIU, HAOYU ZHAO",
    author_email="yqjysczl@gmail.com",
    description="A package for ROOF image processing, YOLO training, and prediction.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Sicyang/AIROOF/tree/master",  # 如果有 GitHub 仓库，填入 URL
    packages=find_packages(),  # 自动发现 `image_processing`, `image_training`, `image_predicting` 等模块
    install_requires=[
        "opencv-python",
        "numpy",
        "pandas",
        "ultralytics",  # YOLO 依赖
        "pillow",  # 处理 PNG 转换
        "fpdf",  # 生成 PDF
        "pypdf2",  # 合并 PDF
        "tqdm"  # 进度条显示
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # 确保 Python 版本兼容
)
