from distutils.core import setup

setup(
    name='kalypso',
    version='0.0.1',
    author='Robert Podschwadt',
    author_email='rpodschwadt1@student.gsu.edu',
    install_requires=[
        "numpy ",
        "sklearn",
        "tensorflow >= 2.3.0",
        "imageio"
    ]
)
