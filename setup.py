from setuptools import setup

setup(
    name='phoenix_simulation',
    version='1.0.0',
    install_requires=[
        'numpy',
        'gym',
        'pybullet',
        'stable-baselines3',
        'torch',
        'scipy>= 1.4'
    ]
)
