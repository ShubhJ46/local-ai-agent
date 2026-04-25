#include <iostream>
#include <vector>
using namespace std;

int startStation(vector<int> &gas, vector<int> &cost) {
    int n = gas.size();
    int startIdx = -1;
    for(int i = 0; i < n; i++) {
        
        // Initially tank is empty
        int currGas = 0;
        bool flag = true;
        for (int j = 0; j < n; j++){
            
            // Circular Index
            int idx = (i + j) % n;
            currGas = currGas + gas[idx] - cost[idx];
            
            // If currGas is less than zero, then it isn't
            // possible to proceed further with this starting point
            if(currGas < 0) {
                flag = false;
                break;  
            }
        }
        
        // If flag is true, then we have found
        // the valid starting point
        if(flag) {
            startIdx = i;
            break;
        }
    }
    return startIdx;
}

int main() {
    vector<int> gas = {4, 5, 7, 4};
    vector<int> cost = {6, 6, 3, 5};
    cout << startStation(gas, cost) << endl;
    return 0;
}