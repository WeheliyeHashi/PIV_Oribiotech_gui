import setuptools

setuptools.setup(
    name="piv_processor",
    version="0.0.1",
    description="Tools to process PIV data",
    long_description="Tools to process PIV data",
    url="",
    author="Weheliye Hashi",
    author_email="Weheliye.Weheliye@oribiotech.com",
    license="MIT",
    packages=setuptools.find_packages(),
    zip_safe=False,
    entry_points={
         'console_scripts': [
            'gui_piv=piv_processor.gui_piv:main',
            'PIV_main_processing=piv_processor.PIV_main_processing:process_main_local',
        ],
    },
)