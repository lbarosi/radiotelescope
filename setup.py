"""Radiotools Module."""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='radiotelescope',
    version='1.0.0',
    description='tools for radioastronomy',
    url='https://github.com/lbarosi/radiotelescope',
    author='Luciano Barosi',
    author_email='lbarosi@df.ufcg.edu.br',
    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Radioastronomy',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='radioastronomy, observation planning, gnuradio',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),  # Required
    python_requires='>=3.7, <4',

    install_requires=['astropy', 'astroquery', 'matplotlib', 'numpy', 'pandas', 'pytz', 'scipy', 'skyfield'],
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/lbarosi/radiotelescope/issues',
        'Source': 'https://github.com/lbarosi/radiotelescope/',
    },
)
