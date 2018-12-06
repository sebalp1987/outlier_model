from setuptools import setup, find_packages

setup(
    name='z_finder',
    version='0.0.1',
    packages=find_packages('z_finder'),
    package_dir={'': 'z_finder'},
    include_package_data=True,
    install_requires=['pandas>=0.19', 'scikit-learn>=0.18', 'numpy', 'seaborn', 'matplotlib', 'imblearn', 'hdbscan',
                      'lightgbm', 'unidecode', 'tqdm', 'beautifulsoup4', 'requests','sphinx', 'networkx', 'docx',
                      'seaborn',  'schedule'],

    python_requires='3.6.1',
    package_data={'': ['*.txt', '*.csv', '*.rst']},


    author="Sebastián Mauricio Palacio",
    author_email="sebastian.mpalacio@gmail.com",
    description="Detecting Insurance Claims Fraud algorithms",
    license="PSF",
    keywords="insurance fraud claim detection machine learning",
    url="sebastian.mpalacio@gmail.com",


)
