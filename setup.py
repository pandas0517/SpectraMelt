from setuptools import setup, find_packages

setup(
    name="spectramelt",              # your package name
    version="1.0.0",
    packages=find_packages(where="src"),  # look for packages inside src/
    package_dir={"": "src"},              # src is the root
    install_requires=[                     # optional
        "numpy",
        "scipy",
        "spgl1"
    ],
    python_requires=">=3.8",
)