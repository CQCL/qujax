from setuptools import setup, find_packages

exec(open('qujax/version.py').read())

setup(
    name="qujax",
    author="Sam Duffield",
    author_email="sam.duffield@quantinuum.com",
    url="https://github.com/CQCL/qujax",
    description="Simulating quantum circuits with JAX",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache 2",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["jax", "jaxlib"],
    classifiers=[
        "Programming Language :: Python",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    include_package_data=True,
    platforms="any",
    version=__version__
)
