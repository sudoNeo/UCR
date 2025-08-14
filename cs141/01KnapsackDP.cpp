class Solution {
  public:
    int knapsack(int W, vector<int> &val, vector<int> &wt) {
        // Define Sub Problen enough for optimal value  
            //Our Knapsack has capacity K with items i to n with corresponding values v_i 
            // We want to find the best total value using items i to n when there is k remaining capacity
            //We want to know the max value per weight =optimal sol
        //Make recc equation to solve
        
            // Item i either contributes or not to the optimal sol
            //if item i is part of my optimal solution to p[i,k] then we know p[i,k]
            //= p[items after i, capacity - weight of i] + the value of i 
            // if item i is not part of my optimal solution then the optimal solution will be 
            //p[items after i, with the same capacity]
            //For the last item n it can only contribute if the capacity left of the knapsack is >= the weight of n
        int n = val.size();
        if(n==0 || W <=0) return 0; 
        //empty Table of sub problems 
        vector<vector<int>> subP(n, vector<int>(W+1,-1));
        
        function<int(int,int)> p =  
        [&]                         
        (int i, int k) -> int       
        {                       
            if (i == n || k == 0) return 0; // 
            if (i == n - 1) return (wt[i] <= k) ? val[i] : 0; 
            int &ans = subP[i][k];     // reference  to sub problem cell
            if (ans != -1) return ans; // subProblem answer already computed
            if (wt[i] > k)             // Weight of I is greater than the current capacity
                return ans = p(i + 1, k);
        
            int take = val[i] + p(i + 1, k - wt[i]); // take item i
            int skip = p(i + 1, k);                  // skip item i
            return ans = max(skip, take);            // store & return best
        };                         

        return p(0,W);
    }
};