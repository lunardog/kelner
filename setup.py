"""
Serves keras models
"""
from setuptools import find_packages, setup

dependencies = [
    'click>=5.0',
    'Keras>=2.0.0',
    'h5py',
    'numpy>=1.0',
    'tensorflow>=1.0',
    'python-magic>=0.4.0',
    'requests',
    'boto3',
    'python-datauri',
    'Pillow>=4.0',
    'requests-mock'
]


setup(
    name='kelner',
    version='0.1.4',
    url='https://github.com/lunardog/kelner',
    license='BSD',
    author='Leszek Rybicki',
    author_email='leszek-rybicki@cookpad.com',
    description='Serve your models',
    long_description=__doc__,
    packages=find_packages(exclude=['tests', 'build', 'dist']),
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    install_requires=dependencies,
    entry_points={
        'console_scripts': [
            'kelner = kelner.cli:kelner',
            'kelnerd = kelner.cli:kelnerd',
        ],
    },
    tests_require=dependencies + ['tempfile', 'moto', 'requests-mock'],
    classifiers=[
        # As from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        # 'Development Status :: 1 - Planning',
        # 'Development Status :: 2 - Pre-Alpha',
        # 'Development Status :: 3 - Alpha',
        'Development Status :: 4 - Beta',
        # 'Development Status :: 5 - Production/Stable',
        # 'Development Status :: 6 - Mature',
        # 'Development Status :: 7 - Inactive',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]
)
