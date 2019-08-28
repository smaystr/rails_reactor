from distutils.core import setup

setup(
    name='Apartment Seller',
    version='0.0.1',
    description='Setup the crawler, which grabs the information from Dom-Ria website.',
    author='Danyil Orel',
    packages=['project'],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
    ],
    requires=['flask', 'sqlalchemy', 'scrapy', 'dotenv']
)
