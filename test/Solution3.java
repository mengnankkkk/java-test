public class Solution3 {
    public int removeElement (int[] nums,int val){
        int ans = 0;
        for(int num:nums){//遍历数组 nums 中的每一个元素
            if(num != val){
                nums[ans] = num;// 将当前元素 num 放入 nums 数组中索引为 ans 的位置
                ans++;
            }
        }
        return ans;
    }
}
