from distutils.core import setup
setup(
  name = 'tfest',
  packages = ['tfest'],
  version = '0.1.1',
  license='MIT',
  description = 'Transfer function estimation based on frequency response.',
  author = 'Giulio Vaccari',
  author_email = 'io@giuliovaccari.it',
  url = 'https://github.com/giuliovv/tfest',
  download_url = 'https://github.com/giuliovv/tfest/archive/refs/tags/v0.1.1-alpha.tar.gz',
  keywords = ['tfest', 'frequency', 'matlab'],
  install_requires=[
          'matplotlib',
          'numpy',
          'scipy'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)