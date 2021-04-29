#include "Astar.h"

using namespace std;

Astar::Astar(double* map, int x_size, int y_size, int z_size, double* start_pose, double* goal_pose) {
    this->map = map;
    this->x_size = x_size;
    this->y_size = y_size;
    this->z_size = z_size;
    this->start_pose = start_pose;
    this->goal_pose = goal_pose;
    this->has_params = 1;
    this->new_map = 1;
    this->goal_pose[2] = 0;
}

Astar::Astar() {
    this->has_params = 0;
}

void Astar::updateParams(double* map, int x_size, int y_size, int z_size, double* start_pose, double* goal_pose) {
    // updates params for instances initialized with Astar() constructor
    this->map = map;
    this->x_size = x_size;
    this->y_size = y_size;
    this->z_size = z_size;
    this->start_pose = start_pose;
    this->goal_pose = goal_pose;
    this->new_map = 1;
    this->has_params = 1;
}

void Astar::updateMap(double* map) {
    // takes in a new map
    this->map = map;
    this->new_map = 1;
}

int Astar::GetMapIndex(int x, int y, int z) {
    // returns map index given (x,y,z) locations
    // assuming 0 start indexing
    int index = z * x_size * y_size;
    index = index + (y * x_size) + x;
    return index;
}

int Astar::IsValidPose(int x, int y, int z) {
    // determine if pose is valid in map
    // check map boundaries
    if (x < 0 || x >= x_size ||
        y < 0 || y >= y_size ||
        z < 0 || z >= z_size) {
        return 0;
    }

    // check the map (0 map value is free, 1 map value is obstacle)
    // for (int i = 0; i < dir; i++) {
    //     if (map[GetMapIndex(int(x+dX[i]), int(y+dY[i]), int(z+dZ[i]))]) {
    //         return 0;
    //     }
    // }
    if (map[GetMapIndex(x, y, z)]) { return 0; }
    return 1;
}

double Astar::Heuristic(Node* node) {
    // // Euclidean Distance
    double dist = 0.0;
    for (int i = 0; i < 2; i++) {
        dist += pow(node->pose[i] - goal_pose[i], 2);
    }

    // Manhattan Distance
    // double dist = 0.0;
    // for (int i = 0; i < numofDOFs; i++) {
    //     dist += abs(node->pose[i] - goal_pose[i]);
    // }
    return weight * dist;
}

double Astar::Cost(Node* node1, Node* node2) {
    // returns euclidean dist (1.4 for diag, 1 for vertical or horizontal)
    if (node1->pose[0] == node2->pose[0] && node1->pose[1] == node2->pose[1]) {
        return this->staycost;
    }
    double cost = 0.0;
    for (int i = 0; i < 2; i++) {
        cost += pow(node1->pose[i] - node2->pose[i], 2);
    }
    return sqrt(cost);
}

void Astar::Expand(Node* node) {
    open.pop();
    int stateindex = GetMapIndex(node->pose[0], node->pose[1], node->pose[2]);
    closed[stateindex] = true;
    int newx, newy, newz, newindex;
    for (int dir = 0; dir < numdir; dir ++) {
        newx = node->pose[0]+dX[dir];
        newy = node->pose[1]+dY[dir];
        newz = node->pose[2]+dZ[dir];
        newindex = GetMapIndex(newx, newy, newz);
        if (!closed[newindex] && (IsValidPose(newx,newy,newz))) {
            if ((newx == this->goal_pose[0]) && (newy == this->goal_pose[1])) {
                if ((newz < this->goal_pose[2]) || (this->goal_pose[2] == 0)) {
                    this->goal_pose[2] = newz;
                    this->goalindex = newindex;
                }
            }
            Node* newnode;
            if (nodemap[newindex] == nullptr) {
                newnode = new Node;
                newnode->pose[0] = newx;
                newnode->pose[1] = newy;
                newnode->pose[2] = newz;
                nodemap[newindex] = newnode;
            }
            else {
                newnode = nodemap[newindex];
            }
            if (newnode->g > (node->g + Cost(node, newnode))) {
                newnode->g = node->g + Cost(node, newnode);
                newnode->h = Heuristic(newnode);
                newnode->f = newnode->g + newnode->h;
                newnode->parent = node;
                open.push(newnode);
            }
        }
    }
}

stack<vector<int>> Astar::MakePlan(Node* startnode, Node* goalnode) {
    stack<Node*> planstack;
    stack<vector<int>> plan; // top is starting node
    Node* curr = goalnode;
    int time = 0;
    while ((curr != startnode) && (curr != nullptr)) {
        while ((curr->pose[0] == curr->parent->pose[0]) && (curr->pose[1] == curr->parent->pose[1])) {
            curr = curr->parent;
        }
        vector<int> temp;
        for (int i = 0; i < numofDOFs; i++) {
            temp.push_back(curr->pose[i]);
        }
        if (!IsValidPose(temp[0],temp[1],temp[2])) {
            cout << "Invalid pose in path at " << time << "." << endl;
        }
        planvec.insert(planvec.begin(),temp);
        this->gplan.push(temp);
        plan.push(temp);
        time ++;
        curr = curr->parent;
    }
    if (curr == startnode) {
        vector<int> temp;
        for (int i = 0; i < numofDOFs; i++) {
            temp.push_back(curr->pose[i]);
        }
        planvec.insert(planvec.begin(),temp);
        this->gplan.push(temp);
        plan.push(temp);
    }
    else {
        cout << "curr != startnode" << endl;
        stack<vector<int>> emptyplan;
        planlength = 0;
        plancost = INFINITY;
        return emptyplan;
    }

    planlength = plan.size();
    plancost = planlength;
    return plan;
}

pair<Astar::Node*, Astar::Node*> Astar::Search() {
    Node* startnode = new Node;
    for (int i = 0; i < numofDOFs; i++) {
        startnode->pose[i] = start_pose[i];
    }
    int startindex = GetMapIndex(startnode->pose[0],startnode->pose[1],startnode->pose[2]);
    startnode->h = Heuristic(startnode);
    startnode->g = 0.0;
    startnode->f = startnode->g + startnode->h;
    startnode->parent = nullptr;
    nodemap[startindex] = startnode;
    open.push(startnode);

    this->goalindex = GetMapIndex(this->goal_pose[0],this->goal_pose[1],this->goal_pose[2]);
    closed[this->goalindex] = false;
    while(!open.empty() && !closed[this->goalindex]) {
        Expand(open.top());
    }
    if (!closed[goalindex]) return make_pair(startnode,startnode);

    return make_pair(startnode, nodemap[this->goalindex]);
}

void Astar::Run() { 
    auto nodepair = Search();
    gplan = MakePlan(nodepair.first, nodepair.second);
    return;
}

stack<vector<int>> Astar::getPlan() {    
    return gplan;
}

vector<vector<int>> Astar::getPlanvec() {
    return planvec;
}

int Astar::getPlanlength() {
    return planlength;
}

double Astar::getPlanCost() {
    return plancost;
}