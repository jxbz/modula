import setuptools

setuptools.setup(
    name="modula",
    packages=setuptools.find_packages(),
    version="0.0.0.0.0.2",
    author="Jeremy Bernstein",
    author_email="jbernstein@mit.edu",
    description="Automatically normalize NN training in the modular norm.",
    url="https://github.com/jxbz/modula",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
            "torch>=2.0.0",
    ],
    python_requires='>=3.9',
)