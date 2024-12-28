from setuptools import setup, find_packages

setup(
    name="processes_image",
    version="0.1.0",
    author="Tuấn Khanh",
    author_email="khanhdevboy@gmail.com",
    description="A simple lib processs image vecto",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NTTKDeveloper/processes_image",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        "matplotlib==3.10.0",
        "numpy==1.24.3",
        "opencv_python==4.8.1.78",
        "opencv_python_headless==4.10.0.84",
        "tqdm==4.66.4"
    ],
)
