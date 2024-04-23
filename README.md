# robotic-powder-weighting

## Getting started

### To build environments
- [install pytorch](https://pytorch.org/get-started/locally/)
- [install isaacgym](https://developer.nvidia.com/isaac-gym)
- install ROS with minimum packages (for simulation)
```
pip install --extra-index-url https://rospypi.github.io/simple rospy-all
pip install --extra-index-url https://rospypi.github.io/simple rosmaster defusedxml
```
- [install ROS (for real-robot)](http://wiki.ros.org/ROS/Installation)

## Experiments
```
cd scoop_env
python scoop.py
```

## Franka Control
```
KEY_UP, "up")
KEY_DOWN, "down")
KEY_LEFT, "left")
KEY_RIGHT, "right")
KEY_W, "backward")
KEY_S, "forward")
KEY_A, "turn_right")
KEY_D, "turn_left")
KEY_E, "turn_up")
KEY_Q, "scoop up")
KEY_E, "scoop down")
KEY_SPACE, "gripper_close")
KEY_X, "save")
KEY_B, "quit")
```

