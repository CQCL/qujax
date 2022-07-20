from setuptools import setup, find_packages

setup(
    name="qujax",
    author="Sam Duffield",
    author_email="sam.duffield@cambridgequantum.com",
    url="https://github.com/CQCL/qujax",
    description="Simulating quantum circuits with JAX",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache 2",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "jax",
        "jaxlib"
    ],
    extras_require={
        "tket": ["pytket", "sympy"],
    },
    classifiers=[
        "Programming Language :: Python",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    include_package_data=True,
    platforms="any",
    version="0.1.0"
)
