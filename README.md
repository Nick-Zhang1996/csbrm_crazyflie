### CSBRM Quadcopter Code
This is the experiment code for running a Crazyflie Quadcopter with various controllers in a Vicon environment. In particular, it is written to test CS-BRM as a controller in Georgia Tech's Indoor Flight Laboratory (MK103). You're welcomed to use the code here for your reference, however we appreciate an email notification. You can find my email in GT directory with name Zhiyuan Zhang. 

### Instructions for using IFL

* Connect power for the cameras and network switch, one is in the northwest corner, the other is near southeast corner. 
* Login the main computer 
* Connect your computer to the LAN port on TP-link router via an ethernet cable
* Connect the lab computer to the LAN port on TP-link router via an ethernet cable
* The DHCP server should be on and assign you an address between 192.168.10.100-199
* If somehow the DHCP server is disabled, you can use a static IP so long as its in the same subnet as the lab computer. Anything from 192.168.10.4-254 should work, subnet mask is 255.255.255.0
* Verify your computer and the lab computer are on the same network, ping each other
* Start the Vicon tracker App on the lab computer, ensure all cameras are online
* You may need to do a calibration
* Use the Active Wand v2 and lay it FLAT on the ground, with the handle pointing south. You may align the origin of the wand (the cross) with the marking on the ground. Ignore the coordinate direction markings written on the ground. 
* The code vicon.py handles coordinate frame transformation. At current configuration it will give you a NED frame. You can use it to offset the origin, read the comment.
* Run vicon.py, you should see your robot state streamed. Triple check that they origin and coordinate axis direction are as expected
* Run Main.py with self.enable_control = False. Go through the program and make sure there are not grammatical mistakes.
* Now change  self.enable_control = True, and proceed with the experiment. 
* Note that Main.py will overwrite the log.p, so backup before you run a new experiment. 
* To read the logs, run planner_analyze.py or analyze.py
