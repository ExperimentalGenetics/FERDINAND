from setuptools import setup, find_packages
import pathlib

def parse_requirements(filename):
    path = pathlib.Path(__file__).parent / filename
    with path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line and not line.startswith("#")]

setup(
    name="ferdinand",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=parse_requirements("requirements.txt"),
    python_requires=">=3.10",
)