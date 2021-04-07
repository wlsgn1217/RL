#!/usr/bin/env python2

import numpy as np

import math
import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from msg_and_srv.srv import RobotControl, RobotControlResponse
from msg_and_srv.srv import RobotReset, RobotResetResponse
from Goal_respawn import Respawn

def main():
    Commander = MoveGroupInterface()
    step = rospy.Service('/robot_control', RobotControl ,Commander.step)
    reset = rospy.Service('/robot_reset', RobotReset, Commander.reset)
    rospy.spin()

def all_close(goal, actual, tolerance):
    if np.linalg.norm(goal-actual) > tolerance:
        return False   
    return True


class MoveGroupInterface(object):
    def __init__(self):
        super(MoveGroupInterface, self).__init__()

         ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group', anonymous=True)

        robot = moveit_commander.RobotCommander()

        group_name = "indy7"
        move_group = moveit_commander.MoveGroupCommander(group_name)
    
        print ("============ Printing robot state ============")
        print (robot.get_current_state())
        print ("")

        self.robot = robot
        self.move_group = move_group

        a = self.reset(req=None)
        

    def reset(self, req):
        ## joint angle, time_step, done, goal initialization
        ## if 'reset' was called by the request, return the initialized state
        try:
            Respawn().delete_model()
        except:
            pass

        initial_point = np.array([0,0,0,0,0,0])
        self.joint_from_current_to_go(initial_point)

        self.get_robot_info()  #update self.joint_angle, pose, current location

        self.time_step = 0
        self.done = False
        self.reward = 0

        self.set_goal()

        print('current time step is', self.time_step)
        #print("Current joint angles are ", self.joint_angle)
        print("Current end-effector pose is ", self.pose, " (xyz, rpy)")
        print("Current goal location is ", self.goal, " (xyz)")
        print("Done is", self.done)

        #joint_angle =self.joint_angle%(math.pi*2.0)
        #print("the joint angles I am sending is ", joint_angle)

        #state = self.joint_angle.tolist() + self.current_location.tolist() + self.goal.tolist()
        #state = joint_angle.round(3).tolist() + self.goal.tolist()
        ee_pose = self.pose
        state = ee_pose.tolist() + self.goal.tolist()
        
        if req == None:
            return
        
        else:

            resp = RobotResetResponse()
            resp.State = state

            return resp

    def step(self, req):
        action = req.Action
        action = np.array(action)
        
        self.get_robot_info()
        target_joint_angle = action + self.joint_angle
        prev_location = self.current_location

        self.joint_from_current_to_go(target_joint_angle)
        #if self.joint_from_current_to_go(target_joint_angle):
        #    self.done = False
        #else:
        #    self.done = True
        self.get_robot_info()
        if all_close(prev_location,self.current_location,0.02): #if it does not move
            penalty = -20.0 #penalty must be a negative number
            print(">>>>>>>>>>Collision Detected: Penalty -20")

        else:
            penalty = 0.0


        #joint_angle = self.joint_angle%(math.pi*2.0)
        #joint_angle = joint_angle.round(3)
        pose = self.pose
        goal = self.goal
        ee_pose = self.pose

        #next_state = joint_angle.tolist() + self.current_location.tolist() + goal.tolist()
        #next_state = joint_angle.tolist() + goal.tolist()
        next_state = ee_pose.tolist() + goal.tolist()

        self.time_step += 1

        if self.time_step > 29:
            self.done = True

        resp = RobotControlResponse()
        resp.State = next_state
        resp.Reward = self.get_reward() + penalty
        resp.Done = self.done

        print('action is ', action)
        print('current time step is', self.time_step)
        #print("Current joint angles are ", self.joint_angle)
        print("Current state is ", next_state, " (xyz)")
        print("Current goal location is ", self.goal, " (xyz)")
        print("Distance is ", self.get_distance())
        print("Done is ", self.done)
        print("Reward is ", resp.Reward)
        #print("the joint angles I am sending is ", joint_angle)

        return resp

    def set_goal(self):
        

        #goal location = 1*3 np.array, [a,b,c] => 0.2<|a,b|<0.6, 0.1<c<0.7
        a_and_b = np.random.choice([1,-1],2)*np.random.uniform(0.2,0.6,size=2)
        c = np.random.uniform(0.1,0.7,1)
        goal = np.append(a_and_b, c).round(3)
        self.goal = goal
        try:
            Respawn().delete_model()
        except:
            pass
        Respawn().respawn_model(goal)

    def get_reward(self):
        distance = self.get_distance()
        

        if all_close(self.goal, self.current_location, 0.05):
            reward = 1000.0
            self.done = True
        else:
            reward = -round(distance**2,3)
            if all_close(self.goal, self.current_location, 0.5):
                print("within the target boundary")
                reward = reward + 5/self.get_distance()

        
        return reward
    
    def get_distance(self):
        distance = np.linalg.norm(self.goal-self.current_location)
        return distance

    def joint_from_current_to_go(self, joint_goal):
        move_group = self.move_group

        move_group.go(joint_goal, wait=True)
        move_group.stop()

        current_joints = move_group.get_current_joint_values()

        #return all_close(joint_goal, current_joints, 0.05)
        

    def get_robot_info(self):
        #Update self.joint angle, pose, current location (type: np.array)

        move_group = self.move_group

        joint_angle = move_group.get_current_joint_values()
        pose = move_group.get_current_pose()
        rpy = move_group.get_current_rpy()

        x = pose.pose.position.x
        y = pose.pose.position.y
        z = pose.pose.position.z
        roll, pitch, yaw = rpy

        #list to array
        joint_angle = np.array(joint_angle)
        pose = np.array([x,y,z,roll,pitch,yaw])
        current_location = np.array([x,y,z])

        #round up
        self.joint_angle = np.round(joint_angle, 3)
        self.pose = np.round(pose, 3)
        self.current_location = np.round(current_location, 3)

    def get_answer(self,req):

        q_goal = np.random.choice([1,-1],6)*np.random.uniform(0,1.5,size=6)
        self.joint_from_current_to_go(q_goal)
        self.get_robot_info
        self.goal = self.current_location
        self.reset(req=None)
        self.get_robot_info

        q_step = q_goal/29

        resp = RobotAnswerResponse()
        resp.Answer = q_step.tolist()

        return resp





if __name__ == '__main__':
    main()