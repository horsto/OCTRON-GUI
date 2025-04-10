from setuptools import setup

# NOTE:
# Weird dependencies on sam-2 are required for the hq-sam-2 package to work.
# This is because the makers of SAM2 HQ have not kept their code up to date with the 
# main SAM2 repository and the package files are overwritten during installation of hq-sam2
# So, we need to reinstall sam2 after hq-sam-2 package is installed.
# This is handled by the post_install.py script.
#
# For pip installations: The post-install hook will run automatically
# For conda installations: Users may need to run 'octron-post-install' manually

setup(
    entry_points={
        'console_scripts': [
            'octron-post-install=octron.post_install:main',
        ],
        'setuptools.installation': [
            'post_install=octron.post_install:run_post_install',
        ],
    },
)
