# Air Keyboard Project
A fully software project that uses a built in webcam and mediapipe to map finger movements in the air to type into an AR Keyboard

# How it works
This project uses google mediapipe models to identify key points on the hand, calculating its coordinates and location in reference to the hand. The project takes a json dataset with data on keyboard keys and parses the data into python objects which are used to build the keys. The project uses depths and coordinates to detect when a finger is pressing on a key, and uses pynput to type the key out. 

# Notes:
This is still a basic prototype for the project so it doesn't work well. Currently coded to only track the index tip keypoint. 
