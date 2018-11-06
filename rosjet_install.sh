if [[ "$(uname -a)" =~ ^.*aarch64.*$ ]]; then
  ISTX1=true
else
  ISTX1=false
fi

# Install Grinch Kernel if Tk1
if ! $ISTX1
then
  cd ~/; git clone https://github.com/jetsonhacks/installGrinch.git
  cd installGrinch; ./installGrinch.sh
fi

#Configure time-zone
sudo dpkg-reconfigure tzdata

#Ros Prerequisites
sudo update-locale LANG=C LANGUAGE=C LC_ALL=C LC_MESSAGES=POSIX
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu trusty main" > /etc/apt/sources.list.d/ros-latest.list'
wget https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -O - | sudo apt-key add -
sudo apt-get update

#Ros kinetic Base
sudo apt-get -y install ros-kinetic-ros-base

#Python Dependencies
sudo apt-get -y install python-rosdep python-dev python-pip python-rosinstall python-wstool

sudo rosdep init
rosdep update
echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

#Ros packages
sudo apt-get -y install ros-kinetic-rosserial-arduino
sudo apt-get -y install ros-kinetic-rosserial
sudo apt-get -y install ros-kinetic-eigen-conversions
sudo apt-get -y install ros-kinetic-tf2-geometry-msgs
sudo apt-get -y install ros-kinetic-angles
sudo apt-get -y install ros-kinetic-web-video-server
sudo apt-get -y install ros-kinetic-rosbridge-suite
sudo apt-get -y install ros-kinetic-rospy-tutorials
sudo apt-get -y install ros-kinetic-joy
sudo apt-get -y install ros-kinetic-teleop-twist-joy
sudo apt-get -y install ros-kinetic-roslint
sudo apt-get -y install ros-kinetic-controller-manager
sudo apt-get -y install ros-kinetic-camera-calibration-parsers
sudo apt-get -y install ros-kinetic-xacro
sudo apt-get -y install ros-kinetic-robot-state-publisher
sudo apt-get -y install ros-kinetic-diff-drive-controller
sudo apt-get -y install ros-kinetic-usb-cam
sudo apt-get -y install ros-kinetic-ros-control
sudo apt-get -y install ros-kinetic-dynamic-reconfigure
sudo apt-get -y install ros-kinetic-fake-localization
sudo apt-get -y install ros-kinetic-joint-state-controller

# Configure Catkin Workspace
source /opt/ros/kinetic/setup.bash
cd ~/catkin_ws/src
catkin_init_workspace

#Install Ros Opencv bindings from source
cd ~/catkin_ws
wstool init src
wstool merge -t src src/rosjet/rosjet.rosinstall
wstool update -t src

#Install Caffe (https://gist.github.com/jetsonhacks/acf63b993b44e1fb9528)
sudo add-apt-repository universe
sudo apt-get update
sudo apt-get install libprotobuf-dev protobuf-compiler gfortran \
libboost-dev cmake libleveldb-dev libsnappy-dev \
libboost-thread-dev libboost-system-dev \
libatlas-base-dev libhdf5-serial-dev libgflags-dev \
libgoogle-glog-dev liblmdb-dev -y
sudo usermod -a -G video $USER
cd ~/
git clone https://github.com/BVLC/caffe.git
cd caffe && git checkout dev
cp Makefile.config.example Makefile.config
make -j 4 all

# System Optimizations
gsettings set org.gnome.settings-daemon.plugins.power button-power shutdown
gsettings set org.gnome.desktop.screensaver lock-enabled false
sudo apt-get -y install compizconfig-settings-manager
gsettings set org.gnome.desktop.interface enable-animations false
gsettings set com.canonical.Unity.Lenses remote-content-search none
echo -e '[SeatDefaults]\nautologin-user=ubuntu' > login_file; sudo mv login_file /etc/lightdm/lightdm.conf
gsettings set org.gnome.Vino enabled true
gsettings set org.gnome.Vino disable-background true
gsettings set org.gnome.Vino prompt-enabled false
gsettings set org.gnome.Vino require-encryption false

echo "alias sr='source ~/catkin_ws/devel/setup.bash'" >> ~/.bashrc

cd ~/catkin_ws
catkin_make && source devel/setup.sh
