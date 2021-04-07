#!/usr/bin/env python2
import rospy
import os
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose

class Respawn():
    def __init__(self):
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('moveit_test/scripts',
                                                'moveit_test/models/goal/model.sdf')
        self.f = open(self.modelPath, 'r')
        self.model = self.f.read()

        self.goal_position = Pose()

    def respawn_model(self,goal_location):
        rospy.wait_for_service('gazebo/spawn_sdf_model')
        x,y,z = goal_location
        self.goal_position.position.x = x
        self.goal_position.position.y = y
        self.goal_position.position.z = z
        spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
        spawn_model_prox('goal', self.model, 'robots_name_space', self.goal_position, "world")



    def delete_model(self):
        rospy.wait_for_service('gazebo/delete_model')
        del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
        del_model_prox('goal')


if __name__ == '__main__':
    r = Respawn()
    r.delete_model()


