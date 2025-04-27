import java.util.Arrays;

class Solution14 {
    public int purchasePlans(int[] nums, int target) {
        int mod = 1_000_000_007;
        int ans = 0;
        Arrays.sort(nums);
        int left = 0;int right = nums.length - 1;
        while (left<right){
            if (nums[left]+nums[right]>target) right--;
            else {
                ans += right - left;
                left++;
            }
            ans %= mod;
        }
        return ans % mod;
    }
}