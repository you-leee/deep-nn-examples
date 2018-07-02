from setuptools import setup

setup(name='Deep Learning Examples',
      version='0.1',
      description='Base Neural Network and Deep learning implementations and examples',
      url='http://github.com/you-leee/deep-learning-examples',
      author='Julia Joosz',
      author_email='joosz.julia@gmail.com',
      license='MIT',
      packages=['dl_examples'],
      install_requires=[
          'numpy',
          'matplotlib',
          'h5py',
          'tensorflow',
          'keras',
          'nltk',
          'pillow',
          'scipy',
      ],
      zip_safe=False)
