import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="shopee",
    version="0.1.0",
    author="namakemono",
    description="Utility tools for kaggle shopee competitions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/namakemono/shopee-toolkit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    license='Apache 2.0',
)

