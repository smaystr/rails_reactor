import setuptools

with open('requirements.txt') as r:
	requirements = r.read().splitlines()

setuptools.setup(
	name='dom_ria_app',
	version='0.0.1',
	install_requires=requirements,
	packages=setuptools.find_packages(),
	url='',
	download_url='',
	description='',
	long_description='',
	license='MIT',
	keywords='',
	classifiers=["Intended Audience :: Developers",
        "Programming Language :: Python",
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
)