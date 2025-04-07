from setuptools import setup, find_packages

setup(
    name="intelligent_control",
    version="0.1.0",
    packages=find_packages(),
    author="booleangabs",
    author_email="gabrieltavares1303@gmail.com",
    description="A package for intelligent control of robotic arm",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/booleangabs/intelligent_control_of_robotic_arm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)