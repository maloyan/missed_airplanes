from setuptools import find_packages, setup

with open("requirements.txt", "r") as fh:
    requirements = fh.readlines()

setup(
    name="missed_planes",
    packages=find_packages(),
    version="0.1.0",
    description="Airplanes classification",
    author="Narek Maloyan",
    license="MIT",
)
