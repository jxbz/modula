import setuptools

setuptools.setup(
    name="modula",
    packages=setuptools.find_packages(),
    version="0.0.1",
    author="Jeremy Bernstein",
    author_email="jbernstein@mit.edu",
    description="Automatically normalize NN training in the modular norm.",
    url="git@github.com:jxbz/modula.git",
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