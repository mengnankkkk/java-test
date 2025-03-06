import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;




class Solution1{
    public int[] twoSum(int[] nums,int target){
        Map<Integer,Integer> map = new HashMap<>();
        for(int i=0;i<nums.length;i++){
            if(map.containsKey(target-nums[i])){
                return new int[] {map.get(target-nums[i]),i};

            }
            map.put(nums[i],i);
        }
        throw new IllegalArgumentException("No two sum solution");
    }
}
class Solution2 {
public List<List<String>> getAnagrams(String[] strs) {
    return new ArrayList<>(Arrays.stream(strs).collect(Collectors.groupingBy(str -> Stream.of(str.split("")).sorted().collect(Collectors.joining()))).values());
}
}
class Solution3{
    public int longestConsecutive(int[] nums){
        int ans = 0;
        Set<Integer> st = new HashSet<>();
        for(int num:nums){
            st.add(num);
        }
        for(int x:st){
            if(st.contains(x-1)){
                continue;
            }
            int y = x+1;
            while(st.contains(y)){
                y++;
            }
            ans = Math.max(ans,y-x);
        }
        return ans;
    }
}
class Solution4 {
    public void moveZeroes(int[] nums){
        if (nums == null){
            return;
        }
        int j =0;
        for (int i=0;i<nums.length;i++){
            if (nums[i]!=0){
                int tmp = nums[i];
                nums[i] = nums[j];
                nums[j++] = tmp;
            }
        }
    }
}
class Solution5{
    public int maxArea(int[] height){
        int res = 0;
        int i = 0;
        int j = height.length-1;
        while (i<j){
            int area = (j-i)*Math.min(height[i],height[j]);
            res = Math.max(res,area);
            if (height[i]<height[j]){
                i++;
            }else {
                j--;
            }
        }
        return res;
    }
}
class Solution6{
    public static List<List<Integer>> threeSum(int[] nums){
        List<List<Integer>> ans = new ArrayList<>();
        int len = nums.length;
        if (nums==null||len<3) return ans;
        Arrays.sort(nums);
        for(int i =0;i<len;i++){
            if (nums[i]>0) break;
            if (i>0&&nums[i]==nums[i-1]) continue;
            int L = i+1;
            int R = len - 1;
            while (L<R){
                int sum = nums[i] + nums[L] + nums[R];
                if (sum == 0){
                    ans.add(Arrays.asList(nums[i],nums[L],nums[R]));
                    while (L<R&&nums[L] == nums[L+1]) L++;
                    while (L<R&&nums[R] == nums[R-1]) R--;
                    L++;
                    R--;
                }
                else if (sum<0) L++;
                else if (sum>0) R--;
            }
        }
        return ans;
    }
}