"""growneuron library setup."""

import pathlib
from setuptools import find_packages
from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')
setup(
    name='growneuron',
    version='0.1',
    description='Gradmax, gradient maximizing neural network growth.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/google-research/growneuron',
    author='Google LLC',
    license='Apache 2.0',
    packages=find_packages(),
    package_data={},
    scripts=['growneuron/cifar/main.py', 'growneuron/imagenet/main.py'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=('neural networks tensorflow machine learning growing growth'
              'gradmax google convolutional during training'),
    install_requires=[
        'absl-py',
        'numpy',
        'ml-collections',
        'tensorflow==2.7',
        'scipy==1.7.3',
        'tfds-nightly',
        ('uncertainty_baselines @ git+https://github.com/google/'
         'uncertainty-baselines.git#egg=uncertainty_baselines'),
    ],
)
