from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="rr-ml-kyrylo-kundik-hw7",
    version="0.0.1",
    install_requires=required,
    packages=["app"],
    url="",
    download_url="",
    description="",
    long_description="",
    license="",
    keywords="",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
)
