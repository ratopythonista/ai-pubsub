# -*- encoding: utf-8 -*-
# Source: https://packaging.python.org/guides/distributing-packages-using-setuptools/

# import io
# import re

from setuptools import find_packages, setup

dev_requirements = ["pylama==7.7.1", "pytest==6.2.4", "black==22.1.0"]
unit_test_requirements = ["pytest==6.2.4", "pytest-cov==2.12.1"]
integration_test_requirements = ["pytest==6.2.4 ", "pytest-cov==2.12.1"]
run_requirements = [
    "celery>=5.2.7",
    "redis>=4.3.3",
    "jsonpickle>=2.2.0",
    "billiard>=3.6.4.0",
    "torch>=1.11.0",
    "torchvision>=0.12.0",
    "eventlet>=0.33.1",
]

# with io.open("__init__.py", encoding="utf8") as version_f:
#     version_match = re.search(
#         r"^__version__ = ['\"]([^'\"]*)['\"]", version_f.read(), re.M
#     )
#     if version_match:
#         version = version_match.group(1)
#     else:
#         raise RuntimeError("Unable to find version string.")

# with io.open("README.md", encoding="utf8") as readme:
#     long_description = readme.read()

setup(
    name="jai-pubsub",
    version="0.0.1",
    author="Rodrigo Guimarães Araújo",
    author_email="rodrigoara27@gmail.com",
    packages=find_packages(exclude="tests"),
    include_package_data=True,
    url="",
    license="COPYRIGHT",
    description="JAI AI PUBSUB",
    # long_description=long_description,
    zip_safe=False,
    install_requires=run_requirements,
    extras_require={
        "dev": dev_requirements,
        "unit": unit_test_requirements,
        "integration": integration_test_requirements,
    },
    python_requires=">=3.9",
    classifiers=[
        "Intended Audience :: Information Technology",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.9",
    ],
)
