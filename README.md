# Install and run
This installation guide is intended for the Ubuntu operating system and has been tested on Ubuntu 20.04.

## Install Docker
https://docs.docker.com/engine/install/

## Install Rocker
```
sudo apt install python3-rocker
```

## Build the Docker image
Open a terminal and the change the directory to `gyro-aided-lucas-kanade`, then run the following command:
```
docker build . -t gyroaidedlucaskanade:latest
```
This step will take a while as it has to download the ROS Docker image and the dataset bag file.

## Run the application in Docker container
Run the following command in a terminal:
```
rocker --x11 gyroaidedlucaskanade:latest /app/build/tracker
```
This should launch windows displaying the image stream from the dataset with tracked features overlaid.