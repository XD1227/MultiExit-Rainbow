MultiExit-Rainbow DQN
======
This is a multi-exit evacuation simulation based on Rainbow DQN, where RVO2 library is used to avoid collision, and Rainbow DQN is applied to guide the pedestrians to evacuate from the indoor room with a more reasonable behavior compared with other mathematical models.

# Keywords
Multi-Exit Evacuation Simulation; Rainbow DQN application
# Setup
## Environment
* Ubuntu 18.04
* CPU: Intel i7 + GPU: Nvidia RTX 2080 ti + RAM: 64G 
* Python 3.7 + Cuda 10.0
## Requirements
* Numpy
* Pytorch 1.0
* OpenCV
* xlwt and xlrd (optional)
# Usage
Directly run the code with default arguments:<br>
'''<br>
python3 main.py
'''<br>
<br>
Or you can be run using the following options:<br>
'''<br>
python3 main.py <br>
--noisy True<br>
--double True<br>
--dueling True<br>
--prioritized_replay True<br>
--c51 True<br>
--multi_step 3<br>
--num_agents 12<br>
--read_model None<br>
--evaluate False<br>
'''<br>
# Demos
There are three demos:
## [different exits width](https://drive.google.com/file/d/19qD0Xqt4rGRUIpeRzEh-MkMhbcEEH-g-/view?usp=sharing)
## [different pedestrian distribution](https://drive.google.com/file/d/1ojG_ggo8m_IkD162QtpIB_djv79yt8IT/view?usp=sharing)
## [different open frame](https://drive.google.com/file/d/1fZHfyebkOTYCXX_dMIj42ju7cPqrj3vu/view?usp=sharing)
# Acknowledgements
- [@Kaixhin](https://github.com/Kaixhin) for [Rainbow DQN](https://github.com/Kaixhin/Rainbow) 
- [@jimfleming](https://github.com/jimfleming) for [RVO2-Python](https://github.com/jimfleming/rvo2)

