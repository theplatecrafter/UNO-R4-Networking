The current program assume that there is an ultrasonic sensor (HC-SR04) and 2 DC motors connected to Arduino UNO R4 Wifi, all mounted on a robot with a phone that acts as the network provider if there isn't any subnet with a valid SSID nearby, and also acts as the camera of the robot.

once this robot has been assembled, run:

Clone into this repository in a directory you want to run the program in
```
git clone https://github.com/theplatecrafter/UNO-R4-Networking
```
Or download the zip file from github and extract it into a directory you want to run the program in.

Then open the folder UNO-R4-Networking in a terminal (Windows open the file explorere to the UNO-R4-Networking folder and type cmd in the address bar and hit enter)
...And run the following command to start:
```
python3 -m launch
```
You will need to go to http://localhost:5000 on your browser.
Follow the instructions shown on the page.