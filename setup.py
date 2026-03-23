from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name= "Machine_efficiency_predictor",
    version= "1.0",
    author= "rohanta_bhamare",
    packages= find_packages(),
    install_requires= requirements

)