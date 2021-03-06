#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math
import numpy as np
from scipy.spatial import KDTree
from std_msgs.msg import Int32

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = .5 


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # ::: Add other member variables
        self.base_waypoints = None
        self.pose = None
        self.stopline_wp_idx = -1
        self.waypoints_2d = None
        self.waypoint_tree = None


        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # ::: Add a subscriber for /traffic_waypoint and /obstacle_waypoint 
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # rospy.spin()
        self.loop()

    def loop(self):
        rate = rospy.Rate(30) # Loop at 50Hz.
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints: # If the pose and base waypoints exist. 
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_id(self):
        x = self.pose.pose.position.x # current pose x coordinate
        y = self.pose.pose.position.y # current pose y coordinate
        nearest_index = self.waypoint_tree.query([x, y], 1)[1] # index of the closest point
        close_coord =  self.waypoints_2d[nearest_index]
        prev_coord = self.waypoints_2d[nearest_index - 1]

        #Equation of the hyperplane passing through close_coord
        cl_vect = np.array(close_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])
        
        # dot product to figure out if the vehicle is in front or behind the vehicles current position.
        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)
        if val > 0:
            nearest_index = (nearest_index + 1) % len(self.waypoints_2d)
        return nearest_index
    
    def publish_waypoints(self):
        # nearest_index = self.get_closest_waypoint_id()
        # lane = Lane()
        lane = self.generate_lane()
        # lane.header = self.base_waypoints.header
        # lane.waypoints = self.base_waypoints.waypoints[nearest_index : nearest_index + LOOKAHEAD_WPS]
        self.final_waypoints_pub.publish(lane)

    def generate_lane(self):
        lane = Lane()
        closest_idx = self.get_closest_waypoint_id()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)
        return lane
    
    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0) # Two waypoints before so front of car is behind line
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.:
                vel = 0.
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        return temp

    def pose_cb(self, msg):
        # ::
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # ::
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            # Get x and y coordinates from the base waypoints.
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] 
                                for waypoint in waypoints.waypoints]
            # Construct a KDTree so the search for the next nearest point can be done in log(n) time.
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # :: Callback for /traffic_waypoint message.
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # :: Callback for /obstacle_waypoint message. 
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
