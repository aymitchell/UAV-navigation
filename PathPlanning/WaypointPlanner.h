#pragma once

#include "Astar.h"

#include <math.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <array>
#include <stack>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <ctime>
#include <numeric>
#include <list>
#include <typeinfo>
#include <memory>
#include <random>
#include <cmath>
#include <variant>
#include <thread>
#include <functional>
#include <utility>

using namespace std;

template <class T> 
class WaypointPlanner {
    public:
        WaypointPlanner(double* map, int x_size, int y_size, int z_size, double* start_pose, vector<double*> waypoints);
        ~WaypointPlanner();
        vector<vector<int>> GetNextWaypointPlan();
        double* GetNextWaypoint();
        vector<vector<vector<int>>> GetFullPlan();
        vector<double*> GetWaypointsOrder();
        vector<int> GetPlanLengths();
        pair<vector<vector<int>>, double*> RunAllPlanners();
    protected:
        double* map;
        int x_size;
        int y_size;
        int z_size;
        double* start_pose;
        static constexpr int numofDOFs = 4; 
        vector<double*> waypoints;
        vector<vector<int>> next_waypoint_plan;
        double* next_waypoint;
        vector<double*> waypoints_order;
        vector<int> plan_lengths;
        int next_waypoint_index;
        double totalcost = 0.0;

        vector<T*> InitPlanners();
        static void RunPlanner(T* planner, int index);
        thread* MakeThread(T* planner);
};