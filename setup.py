from distutils.core import setup
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setup(
  name = 'tfest',
  packages = ['tfest'],
  version = '0.1.3',
  license='MIT',
  description = 'Transfer function estimation based on frequency response.',
  author = 'Giulio Vaccari',
  author_email = 'io@giuliovaccari.it',
  long_description=long_description,
  long_description_content_type="text/markdown",
  url = 'https://github.com/giuliovv/tfest',
  download_url = 'https://github.com/giuliovv/tfest/archive/refs/tags/v0.1.3-alpha.tar.gz',
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
  ],
  python_requires=">=3.6",
)