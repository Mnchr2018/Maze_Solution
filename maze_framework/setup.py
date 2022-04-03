
from setuptools import setup, find_packages

setup(
    name='maze_framework',
    version='1.0.0',
    py_modules=['dntl'],
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='A frameworks python package',
    # long_description=open('README.txt').read(),
    install_requires=['numpy'],
    url='',
    author='',
    author_email=''
)
