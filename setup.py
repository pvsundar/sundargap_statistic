from setuptools import setup, find_packages
import os

# Check if README.md exists and set long_description accordingly
readme_path = 'README.md'
if os.path.exists(readme_path):
    with open(readme_path, 'r') as f:
        long_description = f.read()
else:
    long_description = 'Sundar-Tibshirani Gap Statistic for cluster analysis'

setup(
    name='sundar_gap_stat',
    version='1.2',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'pandas',
        'matplotlib',
        'scipy',
        'openml'  # Added openml for loading datasets like Glass
    ],
    description='Sundar-Tibshirani Gap Statistic for cluster analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='SUNDAR BALAKRISHNAN',
    author_email='pvsundar@gmail.com',
    url='https://github.com/pvsundar/sundar_gap_stat',  # Add your GitHub URL or other URL if available
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
