# Method for Synthetic Dataset Generation of Fused Deposition Modeling (FDM) Parts

This repository is an addon for Blender to simulate a 3D printer.

Current stable version: 0.0.0

## Installation

### Blender

Download Blender installer [here](https://www.blender.org/download/).

### Addon instalation

This repository work as a Blender addon. To use it, you can either install it once - keep in mind that every time you change the code you will have to repeat the addon installation -, or use [VSCode 'Blender Development' extension by Jacques Luke](https://marketplace.visualstudio.com/items?itemName=JacquesLucke.blender-development).

#### VSCode 'Blender Development' extension

Install the extension in VSCode Extensions Marketplace. Then, press `Ctrl/Cmd + P` and search for the `Blender: Start` function. Locate your installed Blender.

After that, you Blender shall start with the extension already loaded. To run the functions in blender, press `F3` and search for the function you want to run. Currently, the main function is `object.node`.

To reload the addon after any change in the code, go back to VSCode and press `Ctrl/Cmd + P` and search for the `Blender: Reload Addons` function. You can now go back to Blender and find the function after pressing `F3`.

#### One time installation

Zip the `addon/` folder and follow the instructions in [here](https://docs.blender.org/manual/en/latest/editors/preferences/addons.html).

## Usage

The addon is composed of functions that can be called from Blender's menu. To open it, go to `Edit > Menu Search...`, and type `fdm_simulator`. The available functions are:

- `fdm_simulator.simulate`: simulates the 3D printing process of the selected object and plot the result in the Blender environment;
- `fdm_simulator.draw_simulation`: draws a previously executed simulation result in the Blender environment;
- `fdm_simulator.scan`: runs a measurement scan on the selected object simulating the behaviour of a laser line scanner and save it as `virtual_scan.npy` in a numpy array format with shape (n, 3), with each row representing (x, y, z) coordinates of the measured points.

It is also possible to execute the simulation outside the Blender environment. It is advantageous because Blender's addons run in a single thread, so the simulation can take a long time to finish. To do so, change the location of the gcode file in main.py and run it. The simulation will be saved in a pickle file to be later used in Blender via `fdm_simulator.draw_simulation` function.

**Important!** - the simulation saves temporary files in the path specified in the `saving_path` argument of the `addon.network.Network` class. Make sure to change it to a valid path in your system. It uses several GBs of disk space.

## Development

It not actually necessary to install python requirements since all the code is supposed to be run on Blender's python and the autoload module is configured to handle installations. But if you want to run any content outside Blender's environment, follow the instructions below.

1. Create and access virstual environment:

on Posix:
``` bash
python -m venv venv && source venv/bin/activate
```

or on Windows:
``` bash
python -m venv venv && venv\\Scripts\\activate.bat
```

2. Install dependencies

``` bash
pip install -r requirements.txt
```

## Changelog

**0.0.0** - 2023-11-29
  - Initial release


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Below is a list of third-party libraries used in this project and their respective licenses:

| Name                      | Version   | License                                                          |
|---------------------------|-----------|------------------------------------------------------------------|
| asttokens                 | 2.2.1     | Apache 2.0                                                       |
| overrides                 | 7.3.1     | Apache License, Version 2.0                                      |
| Cython                    | 0.29.35   | Apache Software License                                          |
| arrow                     | 1.2.3     | Apache Software License                                          |
| bleach                    | 6.0.0     | Apache Software License                                          |
| prometheus-client         | 0.17.0    | Apache Software License                                          |
| q                         | 2.7       | Apache Software License                                          |
| requests                  | 2.31.0    | Apache Software License                                          |
| retrying                  | 1.3.4     | Apache Software License                                          |
| tenacity                  | 8.2.2     | Apache Software License                                          |
| tornado                   | 6.3.2     | Apache Software License                                          |
| tzdata                    | 2023.3    | Apache Software License                                          |
| websocket-client          | 1.6.1     | Apache Software License                                          |
| packaging                 | 23.1      | Apache Software License; BSD License                             |
| python-dateutil           | 2.8.2     | Apache Software License; BSD License                             |
| sniffio                   | 1.3.0     | Apache Software License; MIT License                             |
| jupyterlab-pygments       | 0.2.2     | BSD                                                              |
| matplotlib-inline         | 0.1.6     | BSD 3-Clause                                                     |
| Flask                     | 2.2.5     | BSD License                                                      |
| Jinja2                    | 3.1.2     | BSD License                                                      |
| MarkupSafe                | 2.1.3     | BSD License                                                      |
| Pygments                  | 2.15.1    | BSD License                                                      |
| Send2Trash                | 1.8.2     | BSD License                                                      |
| Werkzeug                  | 2.2.3     | BSD License                                                      |
| appnope                   | 0.1.3     | BSD License                                                      |
| backcall                  | 0.2.0     | BSD License                                                      |
| click                     | 8.1.6     | BSD License                                                      |
| comm                      | 0.1.3     | BSD License                                                      |
| contourpy                 | 1.1.0     | BSD License                                                      |
| cycler                    | 0.11.0    | BSD License                                                      |
| decorator                 | 5.1.1     | BSD License                                                      |
| fastjsonschema            | 2.17.1    | BSD License                                                      |
| idna                      | 3.4       | BSD License                                                      |
| imageio                   | 2.31.3    | BSD License                                                      |
| ipykernel                 | 6.24.0    | BSD License                                                      |
| ipython                   | 8.14.0    | BSD License                                                      |
| ipython-genutils          | 0.2.0     | BSD License                                                      |
| ipywidgets                | 8.1.1     | BSD License                                                      |
| itsdangerous              | 2.1.2     | BSD License                                                      |
| joblib                    | 1.3.1     | BSD License                                                      |
| jsonpointer               | 2.4       | BSD License                                                      |
| jupyter-events            | 0.6.3     | BSD License                                                      |
| jupyter_client            | 8.3.0     | BSD License                                                      |
| jupyter_core              | 5.3.1     | BSD License                                                      |
| jupyter_server            | 2.7.0     | BSD License                                                      |
| jupyter_server_terminals  | 0.4.4     | BSD License                                                      |
| jupyterlab-widgets        | 3.0.9     | BSD License                                                      |
| kiwisolver                | 1.4.4     | BSD License                                                      |
| lazy_loader               | 0.3       | BSD License                                                      |
| memory-profiler           | 0.61.0    | BSD License                                                      |
| mistune                   | 3.0.1     | BSD License                                                      |
| nbclassic                 | 1.0.0     | BSD License                                                      |
| nbclient                  | 0.8.0     | BSD License                                                      |
| nbconvert                 | 7.6.0     | BSD License                                                      |
| nbformat                  | 5.7.0     | BSD License                                                      |
| nest-asyncio              | 1.5.6     | BSD License                                                      |
| networkx                  | 3.1       | BSD License                                                      |
| notebook                  | 6.5.4     | BSD License                                                      |
| notebook_shim             | 0.2.3     | BSD License                                                      |
| numpy                     | 1.25.0    | BSD License                                                      |
| pandas                    | 2.0.2     | BSD License                                                      |
| pandocfilters             | 1.5.0     | BSD License                                                      |
| prettytable               | 3.9.0     | BSD License                                                      |
| prompt-toolkit            | 3.0.39    | BSD License                                                      |
| psutil                    | 5.9.5     | BSD License                                                      |
| pycparser                 | 2.21      | BSD License                                                      |
| python-json-logger        | 2.0.7     | BSD License                                                      |
| scikit-image              | 0.21.0    | BSD License                                                      |
| scikit-learn              | 1.3.0     | BSD License                                                      |
| scipy                     | 1.11.0    | BSD License                                                      |
| seaborn                   | 0.12.2    | BSD License                                                      |
| terminado                 | 0.17.1    | BSD License                                                      |
| threadpoolctl             | 3.2.0     | BSD License                                                      |
| tifffile                  | 2023.8.30 | BSD License                                                      |
| tinycss2                  | 1.2.1     | BSD License                                                      |
| traitlets                 | 5.9.0     | BSD License                                                      |
| webcolors                 | 1.13      | BSD License                                                      |
| webencodings              | 0.5.1     | BSD License                                                      |
| widgetsnbextension        | 4.0.9     | BSD License                                                      |
| zstandard                 | 0.21.0    | BSD License                                                      |
| pyzmq                     | 25.1.0    | BSD License; GNU Library or Lesser General Public License (LGPL) |
| debugpy                   | 1.6.7     | Eclipse Public License 2.0 (EPL-2.0); MIT License                |
| ansi2html                 | 1.8.0     | GNU Lesser General Public License v3 or later (LGPLv3+)          |
| bpy                       | 3.5.0     | GPL-3.0                                                          |
| Pillow                    | 10.0.0    | Historical Permission Notice and Disclaimer (HPND)               |
| isoduration               | 20.11.0   | ISC License (ISCL)                                               |
| pexpect                   | 4.8.0     | ISC License (ISCL)                                               |
| ptyprocess                | 0.7.0     | ISC License (ISCL)                                               |
| dash-core-components      | 2.0.0     | MIT                                                              |
| dash-html-components      | 2.0.0     | MIT                                                              |
| dash-table                | 5.0.0     | MIT                                                              |
| ConfigArgParse            | 1.5.5     | MIT License                                                      |
| PeakUtils                 | 1.3.4     | MIT License                                                      |
| PyWavelets                | 1.4.1     | MIT License                                                      |
| PyYAML                    | 6.0       | MIT License                                                      |
| addict                    | 2.4.0     | MIT License                                                      |
| anyio                     | 3.7.1     | MIT License                                                      |
| argon2-cffi               | 21.3.0    | MIT License                                                      |
| argon2-cffi-bindings      | 21.2.0    | MIT License                                                      |
| attrs                     | 23.1.0    | MIT License                                                      |
| beautifulsoup4            | 4.12.2    | MIT License                                                      |
| cffi                      | 1.15.1    | MIT License                                                      |
| charset-normalizer        | 3.1.0     | MIT License                                                      |
| dash                      | 2.11.1    | MIT License                                                      |
| exceptiongroup            | 1.1.2     | MIT License                                                      |
| executing                 | 1.2.0     | MIT License                                                      |
| fake-bpy-module-3.3       | 20230117  | MIT License                                                      |
| fonttools                 | 4.40.0    | MIT License                                                      |
| jedi                      | 0.18.2    | MIT License                                                      |
| jsonschema                | 4.18.0    | MIT License                                                      |
| jsonschema-specifications | 2023.6.1  | MIT License                                                      |
| open3d                    | 0.17.0    | MIT License                                                      |
| parso                     | 0.8.3     | MIT License                                                      |
| pickleshare               | 0.7.5     | MIT License                                                      |
| pip                       | 23.1.2    | MIT License                                                      |
| pip-licenses              | 4.3.3     | MIT License                                                      |
| platformdirs              | 3.8.1     | MIT License                                                      |
| plotly                    | 5.15.0    | MIT License                                                      |
| pure-eval                 | 0.2.2     | MIT License                                                      |
| pyparsing                 | 3.0.9     | MIT License                                                      |
| pyquaternion              | 0.9.9     | MIT License                                                      |
| pytz                      | 2023.3    | MIT License                                                      |
| referencing               | 0.29.1    | MIT License                                                      |
| rfc3339-validator         | 0.1.4     | MIT License                                                      |
| rfc3986-validator         | 0.1.1     | MIT License                                                      |
| rpds-py                   | 0.8.10    | MIT License                                                      |
| setuptools                | 65.5.0    | MIT License                                                      |
| six                       | 1.16.0    | MIT License                                                      |
| soupsieve                 | 2.4.1     | MIT License                                                      |
| stack-data                | 0.6.2     | MIT License                                                      |
| trimesh                   | 3.22.5    | MIT License                                                      |
| uri-template              | 1.3.0     | MIT License                                                      |
| urllib3                   | 2.0.3     | MIT License                                                      |
| wcwidth                   | 0.2.6     | MIT License                                                      |
| tqdm                      | 4.65.0    | MIT License; Mozilla Public License 2.0 (MPL 2.0)                |
| certifi                   | 2023.5.7  | Mozilla Public License 2.0 (MPL 2.0)                             |
| fqdn                      | 1.5.1     | Mozilla Public License 2.0 (MPL 2.0)                             |
| defusedxml                | 0.7.1     | Python Software Foundation License                               |
| matplotlib                | 3.7.2     | Python Software Foundation License                               |
| typing_extensions         | 4.7.1     | Python Software Foundation License                               |