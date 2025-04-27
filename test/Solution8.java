public class Solution8 {
    public int climbStairs(int n){
        int a = 1,b = 1,sum;
        for(int i = 0;i<n-1;i++){
            sum = a + b;
            a = b;
            b = sum;//a就是n-2，b就是n-1
        }
        return b;
    }
}
