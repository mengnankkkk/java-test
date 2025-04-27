public class Solution12 {
    public int hammingDistance(int x, int y){
       int ans = 0;
       while((x|y)!=0){
           int a = x & 1 ,b = y & 1;
           ans +=a ^b;
           x>>=1;
           y>>=1;
       }
       return ans;
    }
    public  int totalHammingDistance(int[] nums){
        int a = 0;
        int b = 0;
        int ans= 0;
        if(nums!=null){
            for(int num:nums){
                a = num;
            }
            for(int num:nums){
                b = num;
            }
        }
        for(int num:nums) {
            ans += hammingDistance(a, b);
        }
        return ans;
    }
}
