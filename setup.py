import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name             = 'tip-quant',
    version          = '0.1',
    description      = 'Pollen tube quantification method',
    long_description = long_description,
    url              = '',  # Todo: add github url
    author           = 'Quantmetry',
    author_email     = '',  # Todo: add qm mail
    maintainer       = 'Quantmetry',
    packages         = setuptools.find_packages(),
    package_data     = {
        'src': ['config.toml'],
    },
    install_requires = requirements,
    entry_points     = {
        'console_scripts': [
            'tip-quant=src.app:main',
        ],
    },
)
