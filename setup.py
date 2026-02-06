from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(file_path:str)->List[str]:
    """
    Function returns the list of requirements
    """
    requirements=[]
    with open(file_path)  as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]
        requirements = [req for req in requirements if req and not req.startswith("#")]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        
    return requirements

setup(
    name="production-ml-analytics",
    version="0.0.1",
    author="Colin",
    author_email="clngo654@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
