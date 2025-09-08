from setuptools import setup, find_packages

setup(
    name='RegimeDetectionHMM',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy','pandas','matplotlib','hmmlearn','scikit-learn','yfinance','pyyaml'
    ],
    author='Dr. Poulami Nandi',
    description='Hidden Markov Model based market regime detection with Streamlit dashboard and stress tests.',
)
