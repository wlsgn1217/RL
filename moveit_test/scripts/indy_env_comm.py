#!/usr/bin/env python
import random, datetime
import rospy
import std_msgs.msg
import numpy as np
from msg_and_srv.srv import RobotControl, RobotControlRequest
from msg_and_srv.srv import RobotReset, RobotResetRequest
from msg_and_srv.srv import RobotAnswer


class communicate_indy():
    def __init__(self):
        rospy.init_node('Indy_Communication_Client')
        rospy.wait_for_service('/robot_control')
        rospy.wait_for_service('/robot_reset')
        self.step_service = rospy.ServiceProxy('/robot_control', RobotControl, persistent=True)
        self.reset_service = rospy.ServiceProxy('/robot_reset', RobotReset, persistent=True)
        self.get_answer_service = rospy.ServiceProxy('/robot_get_answer', RobotAnswer, persistent=True)
        self.step_object = RobotControlRequest()
         
        
    def reset(self):
        response = self.reset_service()
        obs = np.array([response.State]) ##
       
        return obs

    
    def step(self, action):
        action = action.round(3)
        self.step_object.Action = action[0].tolist()
        response = self.step_service(self.step_object)
        obs_= np.array([response.State])
        reward = response.Reward
        done = response.Done
        
        return obs_, reward, done

    def get_answer(self):
        response = self.get_answer_service()
        answer = np.array(response.Answer)

        return answer