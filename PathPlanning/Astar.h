#pragma once

#include <math.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
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
#include <set>

using namespace std;

class Astar {
    public:
        Astar(double* map, int x_size, int y_size, int z_size, double* start_pose, double* goal_pose);
        Astar();
        void updateParams(double* map, int x_size, int y_size, int z_size, double* start_pose, double* goal_pose);
        void updateMap(double* map);
        void Run();
        stack<vector<int>> getPlan();
        vector<vector<int>> getPlanvec();
        int getPlanlength();
        double getPlanCost();
        void replan(double* start_pose);
        int has_params;

    protected:
        static constexpr int numdir = 9;
        int dX[numdir] = {-1, -1, -1, 0, 0, 1, 1, 1, 0};
        int dY[numdir] = {-1, 0, 1, -1, 1, -1, 0, 1, 0};
        int dZ[numdir] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        int x_size, y_size, z_size;
        double* map;
        double* start_pose;
        double* goal_pose;
        int goalindex;
        int new_map = 0;
        static constexpr int numofDOFs = 3;
        struct Node {
            int pose[numofDOFs];
            double f = INFINITY;
            double g = INFINITY;
            double h = 0.0;
            Node* parent = nullptr;
        };
        struct Compare {
            bool operator()(Node* node1, Node* node2) const {
                return node1->f > node2->f;
            }
        };
        priority_queue<Node*, vector<Node*>, Compare> open;
        unordered_map<int, bool> closed;
        unordered_map<int, Node*> nodemap;
        stack<vector<int>> gplan;
        vector<vector<int>> planvec;
        int planlength;
        double plancost = 0.0;
        double weight = 1.0;
        double staycost = 1.5; // cost to stay in place

        int GetMapIndex(int x, int y, int z);
        int IsValidPose(int x, int y, int z);
        double Heuristic(Node* node);
        double Cost(Node* node1, Node* node2);
        void Expand(Node* node);
        stack<vector<int>> MakePlan(Node* start, Node* goal);
        pair<Node*,Node*> Search();
};