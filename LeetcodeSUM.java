import java.util.*;

public class LeetcodeSUM {

}
class Solution104D{
    public int maxDepth(TreeNode root) {
        if(root==null){
            return 0;
        }
        int left=maxDepth(root.left);
        int right=maxDepth(root.right);
        return Math.max(left,right)+1;
    }
}
class Solution239D{
    public static int[] maxSlidingWindow(int[] nums, int k) {
        if (nums==null||nums.length==0|k==0) return new int[0];
        int n  = nums.length;
        int[] ans = new int[n-k+1];
        Deque<Integer> stack = new ArrayDeque<>();
        for(int i=0;i<n;i++){
            while (!stack.isEmpty()&&nums[stack.getLast()]<=nums[i]){
                stack.removeLast();
            }
            stack.add(i);
            if (stack.peekFirst()<=i-k){
                stack.pollFirst();
            }
            int left = i-k+1;
            if (stack.getFirst()<left){
            stack.removeFirst();
            }
            if (left>=0){
                ans[left] = nums[stack.peekFirst()];
            }
        }
        return ans;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        while (sc.hasNextInt()){
            int n = sc.nextInt();
            int k  = sc.nextInt();
            int[] nums = new int[n];
            for(int i =0;i<n;i++){
                nums[i] = sc.nextInt();
            }
            int[] result = maxSlidingWindow(nums,k);

            StringBuilder  sb = new StringBuilder();
            for (int i=0;i<result.length;i++){
                sb.append(result[i]);
                if (i<result.length-1){
                    sb.append("");
                }
            }
            System.out.printf( sb.toString());
        }
        sc.close();
    }
}
class SolutionMeiTuan1{
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        // 注意 hasNext 和 hasNextLine 的区别
        while (in.hasNextInt()) { // 注意 while 处理多个 case
            int a = in.nextInt();
            int b = in.nextInt();
            System.out.println(a + b);
        }
    }
    public static long solve(int n, int[] a) {
        long[] dp = new long[10];
        Arrays.fill(dp, -1);
        dp[0] = 0; // 初始状态：0个怪物被击败，经验为0

        for (int i = 1; i <= n; i++) {
            long[] next_dp = new long[10];
            Arrays.fill(next_dp, -1);

            for (int j = 0; j < 10; j++) {
                // 决策1: 放走怪物 i
                // 状态从 dp[j] 转移到 next_dp[j]
                if (dp[j] != -1) {
                    long exp = dp[j] + i;
                    next_dp[j] = Math.max(next_dp[j], exp);
                }

                // 决策2: 击败怪物 i
                // 状态从 dp[prev_j] 转移到 next_dp[j]
                int prev_j = (j - 1 + 10) % 10;
                if (dp[prev_j] != -1) {
                    long exp = dp[prev_j] + (long)a[i - 1] * (1 + j);
                    next_dp[j] = Math.max(next_dp[j], exp);
                }
            }
            // 2. dp 更新必须在 j 循环之后！
            dp = next_dp;
        }

        // 3. 计算最终结果和 return 必须在 i 循环之后！
        long max = 0;
        for (long exp : dp) {
            if (exp > max) {
                max = exp;
            }
        }
        return max;
    }
}
class Solution15D{
    public List<List<Integer>> threeSum(int[] nums){
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        for (int k = 0;k<nums.length;k++){
            if (nums[k]>0) break;
            if (k>0&&nums[k]==nums[k-1])continue;
            int i = k+1,j=nums.length-1;
            while (i<j){
                int sum  = nums[k]+nums[i]+nums[j];
                if (sum<0){
                    while (i<j&&nums[i]==nums[++i]);
                }else if (sum>0){
                    while (i<j&&nums[j]==nums[--j]);
                }else {
                    ans.add(new ArrayList<Integer>(Arrays.asList(nums[k],nums[i],nums[j])));
                    while (i<j&&nums[i]==nums[++i]);
                    while (i<j&&nums[j]==nums[--j]);
                }
            }
        }
        return ans;
    }
}
