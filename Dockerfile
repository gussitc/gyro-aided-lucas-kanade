FROM osrf/ros:noetic-desktop-full
WORKDIR /app
RUN apt-get update && apt-get install -y wget

# Get the dataset file
RUN wget http://rpg.ifi.uzh.ch/datasets/uzh-fpv-newer-versions/v3/indoor_forward_7_snapdragon_with_gt.bag
RUN mkdir datasets
RUN mv /app/indoor_forward_7_snapdragon_with_gt.bag /app/datasets/indoor_forward.bag
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    rosbag filter /app/datasets/indoor_forward.bag /app/datasets/indoor_forward_short.bag 't.secs >= 1540821849 and t.secs <= 1540821879'"

# Copy project files
COPY . .

# Build the application
RUN mkdir build
WORKDIR /app/build
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash    && \
                  cmake -D CMAKE_BUILD_TYPE=Release .. && \
                  make"