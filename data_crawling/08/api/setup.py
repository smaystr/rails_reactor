from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="sergey_milantiev_crawler_master",
    version="0.0.0",
    install_requires=requirements,
    packages=["app"],
    author="sergey.milantiev@gmail.com",
    url="",
    download_url="",
    description="CRAWLER DOMRIA API",
    long_description="",
    license="MIT",
    keywords="",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
)
