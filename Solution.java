
import javafx.util.Pair;

import java.util.Arrays;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import java.util.ArrayList;
import java.util.List;


class ListNode {
    int val;
    ListNode next;
    ListNode(int x, ListNode head) { val = x; next = null; }

    public ListNode(int i) {

    }
}

// 1. 两数之和
class Solution1 {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[]{map.get(target - nums[i]), i};
            }
            map.put(nums[i], i);
        }
        throw new IllegalArgumentException("No two sum solution");
    }
}

// 2. 字母异位词分组
class Solution2 {
    public List<List<String>> getAnagrams(String[] strs) {
        return new ArrayList<>(Arrays.stream(strs)
                .collect(Collectors.groupingBy(str -> Stream.of(str.split(""))
                        .sorted().collect(Collectors.joining()))).values());
    }
}

// 3. 最长连续序列
class Solution3 {
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) set.add(num);
        int maxLen = 0;
        for (int num : set) {
            if (!set.contains(num - 1)) {
                int currNum = num, count = 1;
                while (set.contains(currNum + 1)) {
                    currNum++;
                    count++;
                }
                maxLen = Math.max(maxLen, count);
            }
        }
        return maxLen;
    }
}

// 4. 移动零
class Solution4 {
    public void moveZeroes(int[] nums) {
        int j = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                int temp = nums[i];
                nums[i] = nums[j];
                nums[j++] = temp;
            }
        }
    }
}

// 5. 盛最多水的容器
class Solution5 {
    public int maxArea(int[] height) {
        int left = 0, right = height.length - 1, maxArea = 0;
        while (left < right) {
            int area = (right - left) * Math.min(height[left], height[right]);
            maxArea = Math.max(maxArea, area);
            if (height[left] < height[right]) left++;
            else right--;
        }
        return maxArea;
    }
}

// 6. 三数之和
class Solution6 {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int left = i + 1, right = nums.length - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (sum == 0) {
                    res.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    while (left < right && nums[left] == nums[left + 1]) left++;
                    while (left < right && nums[right] == nums[right - 1]) right--;
                    left++; right--;
                } else if (sum < 0) left++;
                else right--;
            }
        }
        return res;
    }
}

// 7. 最长无重复子串
class Solution7 {
    public int lengthOfLongestSubstring(String s) {
        Map<Character, Integer> map = new HashMap<>();
        int left = -1, maxLength = 0;
        for (int right = 0; right < s.length(); right++) {
            if (map.containsKey(s.charAt(right))) {
                left = Math.max(left, map.get(s.charAt(right)));
            }
            map.put(s.charAt(right), right);
            maxLength = Math.max(maxLength, right - left);
        }
        return maxLength;
    }
}

// 8. 链表相交
class Solution8 {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode A = headA, B = headB;
        while (A != B) {
            A = (A == null) ? headB : A.next;
            B = (B == null) ? headA : B.next;
        }
        return A;
    }
}

// 9. 反转链表
class Solution9 {
    public ListNode reverseList(ListNode head) {
        ListNode prev = null, curr = head;
        while (curr != null) {
            ListNode temp = curr.next;
            curr.next = prev;
            prev = curr;
            curr = temp;
        }
        return prev;
    }
}

// 10. 环形链表
class Solution10 {
    public boolean hasCycle(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) return true;
        }
        return false;
    }
}

// 11. 环形链表 II（找到环的入口）
class Solution11 {
    public ListNode detectCycle(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                fast = head;
                while (slow != fast) {
                    slow = slow.next;
                    fast = fast.next;
                }
                return fast;
            }
        }
        return null;
    }
}

// 12. 合并两个有序链表
class Solution12 {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        if (l2 == null) return l1;
        if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }
}


class Solution13 {
            public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
                ListNode pre = new ListNode(0, head);
                ListNode cur = pre;
                int carry = 0;
                while (l1 != null || l2 != null) {
                    int x = l1 == null ? 0 : l1.val;
                    int y = l2 == null ? 0 : l2.val;
                    int sum = x + y + carry;

                    carry = sum / 10;
                    sum = sum % 10;
                    cur.next = new ListNode(sum, head);

                    cur = cur.next;
                    if (l1 != null) {
                        l1 = l1.next;
                    }
                    if (l2 != null) {
                        l2 = l2.next;
                    }
                }
                if (carry == 1) {
                    cur.next = new ListNode(carry, head);
                }
                return pre.next;
            }
        }
        class Solution14 {
            public ListNode removeNthFromEnd(ListNode head, int n) {
                ListNode pre = new ListNode(0, head);
                pre.next = head;
                ListNode start = pre, end = pre;
                while (n != 0) {
                    start = start.next;
                    n--;
                }//start先提前移动
                while (start.next != null) {
                    start = start.next;
                    end = end.next;
                }//一块移动
                end.next = end.next.next;//删除某节点
                return pre.next;
            }
        }
class Solution15 {
    public ListNode swapPairs1(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }//没有节点，或者只剩一个的时候
        ListNode next = head.next;
        head.next = swapPairs1(next.next);//
        next.next = head;//后节点等于头节点
        return next;
    }
    public ListNode swapPairs2(ListNode head){
        ListNode pre = new ListNode(0, head);
        pre.next = head;
        ListNode tmp = pre;
        while (tmp.next!=null&&tmp.next.next!=null){
            ListNode start = tmp.next;
            ListNode end = tmp.next.next;
            tmp.next = end;//head
            start.next = end.next;
            end.next = start;
            tmp = start;
        }
        return pre.next;
    }

}
class Solution16{
    public Node copyRandomList(Node head){
        if (head==null) return null;
        Node cur =head;
        Map<Node,Node> map = new HashMap<>();
        while (cur!=null){
            map.put(cur,new Node(cur.val));
            cur = cur.next;
        }//复制
        cur = head;
        while (cur!=null){
            map.get(cur).next = map.get(cur.next);
            map.get(cur).random = map.get(cur.random);
            cur = cur.next;
        }
        return map.get(head);
    }
}
class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }

    public Node(int val, Node next) {

    }
}
class Solution17{
    public ListNode sortList(ListNode head){
        if (head==null){
            return null;
        }
        ListNode cur = head;
        int n = 0;
        while (cur!=null){
            n++;
            cur = cur.next;
        }
        int[] arr = new int [n];
        cur = head;
        for(int i=0;i<arr.length;i++){
            arr[i] = cur.val;
            cur = cur.next;
        }
        Arrays.sort(arr);
        ListNode listNode = new ListNode(arr[0], head);
        cur = listNode;
        for (int i=1;i<arr.length;i++){
            cur.next = new ListNode(arr[i], head);
            cur = cur.next;
        }
        return listNode;
    }
}
class LRUCache extends LinkedHashMap<Integer,Integer> {
    private int capacity;
    public LRUCache(int capacity) {
        super(capacity,0.75F,true);
        this.capacity = capacity;
    }
    public int get(int key) {
        return super.getOrDefault(key,-1);
    }
    public void put(int key, int value) {
        super.put(key,value);
    }
    @Override
    protected boolean removeEldestEntry(Map.Entry<Integer,Integer> eldest){
        return size() > capacity;
    }
}
class Solution18{
    public int maxSubArray(int[] nums){
        int ans = nums[0];
        int sum = 0;
        for (int num:nums){
            if (sum>0){
                sum+=num;
            }
            else {
                sum = num;
            }
            ans=Math.max(ans,sum);
        }
        return ans;
    }
}
class Solution19{
    public int[][] merge(int[][] intervals){
        Arrays.sort(intervals,(p,q)->p[0] - q[0]);//按照左端点从小到大排序
        List<int[]> ans = new ArrayList<>();
        for (int[] p:intervals){
            int m = ans.size();
            if (m>0&&p[0]<=ans.get(m-1)[1]){//限制范围
                ans.get(m-1)[1] = Math.max(ans.get(m-1)[1],p[1]);
            }
            else{
                ans.add(p);
            }
        }
        return ans.toArray(new int[ans.size()][]);

    }
}
class Solution20{
    public void rotate(int[] nums,int k){
        k%=nums.length;
        reverse(nums,0,nums.length-1);
        reverse(nums,0,k-1);
        reverse(nums,k,nums.length-1);

    }
    public void reverse(int[] nums,int start,int end){
        while (start<end){
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start+=1;
            end-=1;
        }
    }
}
class Solution21{
    public int[] productExcepSelf(int[] nums){
        int len = nums.length;
        if (len==0) return new  int[0];
        int[] ans = new int[len];
        ans[0] = 1;
        int tmp = 1;
        for (int i=1;i<len;i++){
            ans[i] = ans[i-1] * nums[i-1];
        }
        for (int i=len-2;i>=0;i--){
            tmp *=nums[i+1];
            ans[i] *= tmp;
        }
        return ans;
    }
}
class TreeNode{
    int val;
    TreeNode left;
    TreeNode right;

    public TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }

    public TreeNode(int val) {
        this.val = val;
    }
    public TreeNode(){}
}
class Solution22{
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        dfs(res, root);
        return res;
    }
        void dfs(List<Integer> res,TreeNode root){
            if (root==null){
                return ;
            }
            dfs(res,root.left);
            res.add(root.val);
            dfs(res,root.right);
    }
}
class Solution23{
    public int maxDepth(TreeNode root){
        if (root==null){
            return 0;
        }
        else {
            int left = maxDepth(root.left);
            int right = maxDepth(root.right);
            return Math.max(left,right)+1;
        }
    }
}
class Solution24{
    public TreeNode invertTree(TreeNode root){
        if (root==null) return null;
        TreeNode tmp = root.left;
        root.left = invertTree(root.right);
        root.right = invertTree(tmp);
        return root;
    }
}
class Solution25{
    public boolean isSymmetric(TreeNode root){
        return root==null||recur(root.left,root.right);
    }
    boolean recur(TreeNode L,TreeNode R){
        if (L==null&&R==null) return true;
        if (L==null||R==null||L.val!=R.val) return false;
        return recur(L.left,R.right)&&recur(L.right,R.left);
    }
}
class Solution26{
    private int ans;

    public int diameterOfBinaryTree(TreeNode root){
        dfs(root);
        return ans;
    }
    private int dfs(TreeNode node){
        if (node ==null){
            return -1;
        }
        int llen = dfs(node.left)+1;
        int rlen = dfs(node.right)+1;
        ans = Math.max(ans,llen+rlen);
        return Math.max(llen,rlen);
    }
}
class Solution27{
    public List<List<Integer>> levelOrder(TreeNode root){
        List<List<Integer>> res = new ArrayList<>();
        if (root==null) return res;
        Queue<TreeNode> que = new LinkedList<>();
        que.add(root);
        while (!que.isEmpty()){
            int sized = que.size();
            List<Integer> layer = new ArrayList<>();
            while (sized-->0){
                TreeNode poll = que.poll();
                layer.add(poll.val);
                if (poll.left!=null) que.add(poll.left);
                if (poll.right!=null) que.add(poll.right);
            }
            res.add(layer);
        }return res;
    }
}
class Solution28{
    public TreeNode sortedArrayToBST(int[] nums){
        return dfs(nums,0,nums.length-1);
    }
    private TreeNode dfs(int[] nums,int lo,int hi){
        if (lo>hi){
            return null;
        }
        int mid = lo+(hi-lo)/2;
        TreeNode root  = new TreeNode(nums[mid]);
        root.left = dfs(nums,lo,mid-1);
        root.right = dfs(nums,mid+1,hi);
        return root;
    }
}
class Solution29{
    public boolean isValidBST(TreeNode root){
        return isValidBST(root,Long.MIN_VALUE,Long.MAX_VALUE);
    }
    private boolean isValidBST(TreeNode node,long left,long right){
        if (node==null){
            return true;
        }
        long x = node.val;
        return left<x&&x<right&&
                isValidBST(node.left,left,x)&&
                isValidBST(node.right,x,right);
    }
}
class Solution30{
    int res,k;
    void dfs(TreeNode root){
        if (root==null) return;
        dfs(root.left);
        if (k==0) return;
        if (--k==0) res = root.val;
        dfs(root.right);
    }
    public int kthSmallest(TreeNode root,int k){
        this.k = k;
        dfs(root);
        return res;
    }
}
class Solution31{
    public List<Integer> rightSideView(TreeNode  root){
        List<Integer> ans = new ArrayList<>();
        dfs(root,0,ans);
        return ans;
    }
    private void dfs(TreeNode root,int depth,List<Integer> ans){
        if (root==null){
            return;
        }
        if (depth==ans.size()){
            ans.add(root.val);
        }
        dfs(root.right,depth+1,ans);
        dfs(root.left,depth+1,ans);
    }
}
class Solution32 {
    public void flatten(TreeNode root) {
        while (root != null) {
            if (root.left != null) {
                TreeNode pre = root.left;
                while (pre.right != null) { // 找到左子树的最右节点
                    pre = pre.right;
                }
                pre.right = root.right; // 右子树接到左子树的最右节点上
                root.right = root.left; // 左子树变成右子树
                root.left = null; // 断开左子树
            }
            root = root.right; // 继续处理下一个节点
        }
    }
}
class Solution33{
    private int ans;
    public int pathSum(TreeNode root, int targetSum) {
        Map<Long, Integer> cnt = new HashMap<>();
        cnt.put(0L, 1);
        dfs(root, 0, targetSum, cnt);
        return ans;
    }
    private void dfs(TreeNode node,long s,int targetsum,Map<Long,Integer> cnt){
        if (node==null){
            return;
        }
        s+= node.val;
        ans+=cnt.getOrDefault(s-targetsum,0);
        cnt.merge(s,1,Integer::sum);//cnt[s++]
        dfs(node.left,s,targetsum,cnt);
        dfs(node.right,s,targetsum,cnt);
        cnt.merge(s,-1,Integer::sum);//归零
    }
}
class Solution34{
    public TreeNode lowestCommonAncestor(TreeNode root,TreeNode p,TreeNode q){
        if (root==null||root==p||root==q) return root;
        TreeNode left = lowestCommonAncestor(root.left,p,q);
        TreeNode right = lowestCommonAncestor(root.right,p,q);
        if (left==null) return right;
        if (right==null) return left;
        return root;
    }
}
class Solution35{
    int[] preorder;
    HashMap<Integer,Integer> dic = new HashMap<>();
    public TreeNode bulidTree1(int[] preorder,int[] inorder){
        this.preorder =  preorder;
        for (int i=0;i<inorder.length;i++)
            dic.put(inorder[i],i);
        return recur(0,0,inorder.length-1);
    }
    TreeNode recur(int root,int left,int right){
        if (left>right) return null;
        TreeNode node = new TreeNode(preorder[root]);
        int i = dic.get(preorder[root]);
        node.left = recur(root+1,left,i-1);
        node.right = recur(root+i-left+1,i+1,right);
        return node;
    }

    /**
     * 解法2
     */
    int pre = 0;
    int in=0;
    public TreeNode buildTree2(int[] preorder,int[] inorder){
        return my(preorder,inorder,Integer.MAX_VALUE);
    }
    public TreeNode my(int[] preorder,int[] inorder,int stop){
        if (pre == preorder.length){
            return null;
        }
        if (inorder[in] ==stop){
            in++;
            return null;
        }
        TreeNode root = new TreeNode(preorder[pre++]);
        root.left = my(preorder,inorder,root.val);
        root.right = my(preorder,inorder,stop);
        return root;
    }
}
class Solution36{
    public int searchInsert(int[] nums,int target){
        int left = 0,right = nums.length-1;
        while (left<=right){
            int mid = (left+right)/2;
            if (nums[mid] == target){
                return mid;
            }
            else if (nums[mid]<target){
                left =mid+1;
            }
            else {
                right = mid-1;
            }
        }
        return left;
    }
}
class Solution37{
    public boolean searchMatrix(int[][] matrix,int target){
        int m = matrix.length;
        int n = matrix[0].length;
        int left = -1;
        int right = m*n;
        while (left+1<right){
            int mid =(left+right)>>>1;
            int x = matrix[mid/n][mid%n];
            if (x==target){
                return true;
            }
            if (x<target){
                left = mid;
            }
            else {
                right = mid;
            }
        }
        return false;
    }
}
class Solution38{
    public int[] searchRange(int[] nums,int target){
        int start = lowBound(nums,target);
        if (start==nums.length||nums[start] !=target){
            return new int[] {-1,-1};
        }
        int end = lowBound(nums,target+1)-1;
        return new int[] {start,end};
    }
    private int lowBound(int[] nums,int target){
        int left = -1;
        int right = nums.length;
        while (left+1<right){
            int mid = left+(right-left)/2;
            if (nums[mid]>=target){
                right = mid;
            }else {
                left = mid;
            }
        }
        return right;
    }
}
class Solution39{
    public int search(int[] nums,int target){
        if (nums==null||nums.length==0){
            return -1;
        }
        int start = 0;
        int end = nums.length-1;
        int mid;
        while (start<=end){
            mid = start+(end-start)/2;
            if (nums[mid] ==target){
                return mid;
            }
            if (nums[start]<=nums[mid]){
                if (target>=nums[start]&&target<nums[mid]){
                    end  = mid-1;
                }else {
                    start = mid+1;
                }
            }else {
                if (target<=nums[end]&&target>nums[mid]){
                    start = mid +1;
                }
                else {
                    end = mid-1;
                }
            }
        }
        return -1;
    }
}
class Solution40{
    public int findMin(int[] nums){
        int n = nums.length;
        int left  = -1;
        int right = n-1;
        while (left+1<right){
            int mid = (left+right)>>>1;
            if (nums[mid]<nums[n-1]){
                right = mid;
            }
            else {
                left = mid;
            }
        }
        return nums[right];
    }
}
class Solution41{
    public List<Integer> findAnagrams(String s,String p) {
        List<Integer> ans = new ArrayList<>();
        int[] cnt = new int[26];
        for (char c : p.toCharArray()) {
            cnt[c - 'a']++;
        }
        int left = 0,right=0,required = p.length();
        while (right<s.length()){
            int c = s.charAt(right) -'a';
            if (cnt[c]>0){
                required--;
            }
            cnt[c]--;
            right++;
            if (required==0){
                ans.add(left);
            }
            if (right-left==p.length()){
                int l = s.charAt(left)-'a';
                if (cnt[l]>=0){
                    required++;
                }
                cnt[l]++;
                left++;
            }
        }
        return ans;
    }
}
class Solution42{
    public void setZeroes(int[][] matrix){
        Set<Integer> row_zero = new HashSet<>();
        Set<Integer> col_zero = new HashSet<>();
        int row = matrix.length;
        int col = matrix[0].length;
        for (int i =0;i<row;i++){
            for (int j =0;j<col;j++){
                if (matrix[i][j]==0){
                    row_zero.add(i);
                    col_zero.add(j);
                }
            }
        }
        for (int i =0;i<row;i++){
            for (int j =0;j<col;j++){
                if (row_zero.contains(i)||col_zero.contains(j)){
                    matrix[i][j] = 0;
                }
            }
        }
    }
}

class Solution43 {
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<>();
        if (matrix.length == 0) return res;

        int l = 0, r = matrix[0].length - 1;
        int t = 0, b = matrix.length - 1;

        while (l <= r && t <= b) {
            // 从左到右
            for (int i = l; i <= r; i++) res.add(matrix[t][i]);
            t++;  // 更新上边界
            if (t > b) break;

            // 从上到下
            for (int i = t; i <= b; i++) res.add(matrix[i][r]);
            r--;  // 更新右边界
            if (l > r) break;

            // 从右到左
            for (int i = r; i >= l; i--) res.add(matrix[b][i]);
            b--;  // 更新下边界
            if (t > b) break;

            // 从下到上
            for (int i = b; i >= t; i--) res.add(matrix[i][l]);
            l++;  // 更新左边界
            if (l > r) break;
        }

        return res;
    }
}
class Solution44{
    public void rotate(int[][] matrix){
        int n = matrix.length;
        int[][] matrix_new  = new int[n][n];
        for (int i =0;i<n;++i){
            for (int j =0;j<n;j++){
                matrix_new[j][n-i-1] = matrix[i][j];
            }
        }
        for (int i=0;i<n;++i){
            for (int j=0;j<n;++j){
                matrix[i][j] = matrix_new[i][j];
            }
        }
    }
}
class Solution45{
    public boolean searchMatrix(int[][] matrix,int target){
        int i = matrix.length-1,j=0;
        while (i>=0&&j<matrix[0].length){
            if (matrix[i][j]>target) i--;
            else if (matrix[i][j]<target) j++;
            else return true;
        }
        return false;
    }
}
class Solution46{
    public int subarraySum(int[] nums,int k){
        int n = nums.length;
        int [] s = new int [n+1];
        for (int i =0;i<n;i++){
            s[i+1] = s[i] + nums[i];
        }
        int ans= 0;
        Map<Integer,Integer> cnt = new HashMap<>(n+1);
        for (int sj:s){
            ans+=cnt.getOrDefault(sj-k,0);
            cnt.merge(sj,1,Integer::sum);//cnt[sj]++
        }
        return ans;
    }
}
class Solution47{
    public int numIslands(char[][] grid){
        int count = 0;
        for (int i =0;i<grid.length;i++){
            for (int j=0;j<grid[0].length;j++){
                if (grid[i][j] =='1'){
                    dfs(grid,i,j);
                    count++;
                }
            }
        }
        return count;
    }
    private void dfs(char[][] grid,int i,int j){
        if (i<0||j<0||i>=grid.length||j>=grid[0].length||grid[i][j]=='0') return;
        grid[i][j]='0';
        dfs(grid,i+1,j);
        dfs(grid,i,j+1);
        dfs(grid,i-1,j);
        dfs(grid,i,j-1);

    }
}

class Solution48 {
    public int orangesRotting(int[][] grid) {
        int[][] dir = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        int m = grid.length;
        int n = grid[0].length;
        Queue<int[]> queue = new LinkedList<>();
        int freshCount = 0;

        // 统计新鲜橘子数量，并找到腐烂橘子的初始位置
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {  // 修正错误 j < 0
                if (grid[i][j] == 2) {
                    queue.offer(new int[]{i, j});
                } else if (grid[i][j] == 1) {
                    freshCount++;
                }
            }
        }

        // 没有新鲜橘子，直接返回 0
        if (freshCount == 0) {
            return 0;
        }

        int time = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            boolean hasRotten = false;
            for (int i = 0; i < size; i++) {
                int[] arr = queue.poll();
                int x = arr[0];
                int y = arr[1];

                // 遍历 4 个方向
                for (int j = 0; j < 4; j++) {
                    int xNext = x + dir[j][0];
                    int yNext = y + dir[j][1];  // 修正错误：应使用 dir[j][1]

                    // 检查边界条件 & 是否是新鲜橘子
                    if (xNext >= 0 && yNext >= 0 && xNext < m && yNext < n && grid[xNext][yNext] == 1) {
                        grid[xNext][yNext] = 2;
                        queue.offer(new int[]{xNext, yNext});
                        freshCount--;
                        hasRotten = true;
                    }
                }
            }
            // 只有在本轮有橘子腐烂时，才增加时间
            if (hasRotten) {
                time++;
            }
        }

        // 如果还有新鲜橘子，返回 -1
        return freshCount == 0 ? time : -1;
    }
}
class Solution49{
    public boolean canFinsh(int numCourses,int [][] prerequisites){
        int[] indegress  = new int[numCourses];
        List<List<Integer>> adjacency = new ArrayList<>();
        Queue<Integer> queue = new LinkedList<>();
        for (int i =0;i<numCourses;i++){
            adjacency.add(new ArrayList<>());
        }
        for (int[] cp:prerequisites){
            indegress[cp[0]]++;
            adjacency.get(cp[1]).add(cp[0]);
        }
        for (int i =0;i<numCourses;i++){
            if (indegress[i]==0) queue.add(i);
        }
        while (!queue.isEmpty()){
            int pre  = queue.poll();
            numCourses--;
            for (int cur:adjacency.get(pre))
                if (--indegress[cur]==0) queue.add(cur);
        }
        return numCourses==0;
    }
}
class Solution50{
    List<Integer> nums;
    List<List<Integer>> res;
    void swap(int a,int b){
        int tmp = nums.get(a);
        nums.set(a,nums.get(b));
        nums.set(b,tmp);
    }
    void dfs(int x){
        if (x==nums.size()-1){
            res.add(new ArrayList<>(nums));
            return;
        }
        for (int i =x;i<nums.size();i++){
            swap(i,x);
            dfs(x+1);
            swap(i,x);
        }
    }
    public List<List<Integer>> permute(int[] nums){
        this.res = new ArrayList<>();
        this.nums = new ArrayList<>();
        for (int num:nums){
            this.nums.add(num);
        }
        dfs(0);
        return res;
    }
}
class Solution51{
    public List<List<Integer>> subsets(int[] nums){
        List<List<Integer>> res = new ArrayList<>();
        dfs(nums,new ArrayList<>(),0,res);
        return res;
    }
    public void dfs(int[] nums,List<Integer> row,int n,List<List<Integer>> res){
        if (n ==nums.length){
            res.add(new ArrayList<>(row));
            return;
        }
        int nthNumber = nums[n];
        dfs(nums,row,n+1,res);
        row.add(nthNumber);
        dfs(nums,row,n+1,res);
        row.remove(row.size()-1);
    }
}
class Solution52{
    private String letterMap[] = {
            " ",    //0
            "",     //1
            "abc",  //2
            "def",  //3
            "ghi",  //4
            "jkl",  //5
            "mno",  //6
            "pqrs", //7
            "tuv",  //8
            "wxyz"  //9
    };
    private ArrayList<String> res;

    public List<String> letterCombinations(String digits){
        res = new ArrayList<String>();
        if (digits.equals("")){
            return res;
        }
        findCombination(digits,0,"");
        return res;
    }
    private void findCombination(String digits,int index,String s){
        if (index==digits.length()){
            res.add(s);
            return;
        }
        Character c = digits.charAt(index);
        String letters = letterMap[c-'0'];
        for (int i =0;i<letters.length();i++){
            findCombination(digits,index+1,s+letters.charAt(i));

        }
        return;
    }
}
class Solution53{
    void backtrack(List<Integer> state,int target,int[] choices,int start,List<List<Integer>> res){
        if (target==0){
            res.add(new ArrayList<>(state));
            return;
        }
        for (int i=start; i<choices.length;i++){
            if (target-choices[i]<0){
                break;
            }
            state.add(choices[i]);
            backtrack(state,target-choices[i],choices,i,res);
            state.remove(state.size()-1);
        }
    }
    public List<List<Integer>> combinationSum(int[] candidates,int target){
        List<Integer> state = new ArrayList<>();
        Arrays.sort(candidates);
        int start = 0;
        List<List<Integer>> res = new ArrayList<>();
        backtrack(state,target,candidates,start,res);
        return res;
    }
}
class Solution54{
    private int n;
    private final List<String> ans = new ArrayList<>();
    private char[] path;

    public List<String> generateParenthesis(int n){
        this.n = n;
        path = new char[n*2];
        dfs(0,0);
        return ans;
    }


    private void dfs(int i ,int open){
        if (i==n*2){
            ans.add(new String(path));
            return;
        }
        if (open<n){
            path[i] = '(';
            dfs(i+1,open+1);
        }
        if (i-open<open){
            path[i] = ')';
            dfs(i+1,open);
        }
    }
}
class Solution55{
    static int[][] points = new int[][]{{1, 0}, {-1, 0}, {0, -1}, {0, 1}};

    public boolean exist(char[][] board,String word){
        if (board.length==0){
            return false;
        }
        char[] chars =word.toCharArray();
        for (int i =0;i<board.length;i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (dfs(i, j, board, 0, chars)) {
                    ;
                    return true;
                }
            }
        }

        return false;
    }
    private  boolean dfs(int x,int y,char[][] board,int index,char[] chars){
        if (x<0||x>board.length-1||
        y<0||y>board[0].length-1||
                board[x][y]!=chars[index]
        ){
            return false;
        }
        if (index==chars.length-1){
            return true;
        }
        board[x][y] ='\0';
        for (int i =0;i<4;i++){
            if (dfs(x+points[i][0],y+points[i][1],board,index+1,chars)){
                return true;
            }
        }
        board[x][y] =chars[index];
        return false;
    }
}
class Solution56{
    private final List<List<String>> ans = new ArrayList<>();
    private final List<String> path = new ArrayList<>();
    private String s;

    public List<List<String>> partition(String s) {
        this.s = s;
        dfs(0);
        return ans;
    }

    private void dfs(int i){
        if (i==s.length()){//分割完毕
            ans.add(new ArrayList<>(path));
            return;
        }
        for (int j=i;j<s.length();j++){//枚举结束的位置
            if (isPalindrom(i,j)){
                path.add(s.substring(i,j+1));//分割
                dfs(j+1);
                path.remove(path.size()-1);//回溯
            }
        }
    }
    private boolean isPalindrom(int left,int right){
        while (left<right){
            if (s.charAt(left++)!=s.charAt(right--)){
                return false;
            }
        }
        return true;
    }
}
class Solution57{
    public boolean isValid(String s){
        if (s.isEmpty())
            return true;
        Stack<Character> stack = new Stack<Character>();
        for (char c:s.toCharArray()){
            if (c=='(')
                stack.push(')');
            else if (c=='{')
                stack.push('}');
            else if (c=='[')
                stack.push(']');
            else if (stack.empty()||c!=stack.pop())
                return false;
        }
       return stack.empty();
    }
}
class Solution101{
    public long mostPoints(int[][] questions){
        long[] memo = new long[questions.length];
        return dfs(0,questions,memo);
    }
    private long dfs(int i,int[][] questions,long[] memo){
        if (i>=memo.length){
            return 0;
        }
        if (memo[i]>0){
            return memo[i];
        }
        long notChoose  = dfs(i+1,questions,memo);
        long choose = dfs(i+questions[i][1]+1,questions,memo)+questions[i][0];
        return memo[i] = Math.max(notChoose,choose);
    }
}
class Solution58{
    private Stack<Integer> stack;
    private Stack<Integer> min_stack;
    public void setMin_stack(){
        stack = new Stack<>();
        min_stack = new Stack<>();
    }
    public void push(int x){
        stack.push(x);
        if (min_stack.isEmpty()||x<=min_stack.peek())
            min_stack.push(x);
    }
    public void pop(){
        if (stack.pop().equals(min_stack.peek()))
            min_stack.pop();
    }
    public int top(){
        return stack.peek();
    }
    public int getMin(){
        return min_stack.peek();
    }
}
class Solution59{
    public String decodeString(String s){
        Stack<Integer> countstack = new Stack<>();
        Stack<StringBuilder> stringstack = new Stack<>();
        StringBuilder currentString = new StringBuilder();
        int k = 0;//重复次数


        for (char c:s.toCharArray()){
            if (Character.isDigit(c)){//如果c为数字的话
                k = k*10+(c-'0');//多位数字
            }else if (c=='['){//开始
                countstack.push(k);
                k=0;
                stringstack.push(currentString);
                currentString = new StringBuilder();
            }else if (c==']'){//结束
                int repeat = countstack.pop();
                StringBuilder sb = stringstack.pop();
                for (int i =0;i<repeat;i++){
                    sb.append(currentString);//
                }//开始重复
                currentString = sb;//更新拼接之后的字符串
            }else {
                currentString.append(c);
            }
        }
        return currentString.toString();
    }
}
class Solution102{
    public long maximumTripletValue(int[] nums){
        long ans = 0,mxDiff = 0;
        int mx = 0;
        for (int x:nums){
            ans = Math.max(ans,mxDiff*x);
            mxDiff = Math.max(mx-x,mxDiff);
            mx  = Math.max(mx,x);
        }
        return ans;
    }
}
class Solution60{
    public int[] dailyTemperatures1(int[] T) {
        int length = T.length;
        int[] result = new int[length];

        for (int i = length - 2; i >= 0; i--) {
            for (int j = i + 1; j < length; j += result[j]) {
                if (T[j] > T[i]) {
                    result[i] = j - i;
                    break;
                } else if (result[j] == 0) {
                    result[i] = 0;
                    break;

                }
            }
        }
        return result;
    }
    public int[] dailyTemperatures(int[] temperatures){
        int n  = temperatures.length;
        int[] ans = new int[n];

        Deque<Integer> st  = new ArrayDeque<>();

        for (int i =n-1;i>=0;i--){
            int t = temperatures[i];
            while (!st.isEmpty()&&t>=temperatures[st.peek()]){
                st.pop();
            }
            if (!st.isEmpty()){
                ans[i] = st.peek()-i;
            }
            st.push(i);
        }
        return ans;

    }
}
class Solution61{
    public int findKthLargest(int[] nums,int k){
        Arrays.sort(nums);
        return nums[nums.length-k];
    }
}
class Solution62{
    public int[] topKFrequent(int[] nums,int k){
        // 统计每个数字出现的次数
        Map<Integer,Integer> counter = IntStream.of(nums).boxed().collect(Collectors.toMap(e->e,e->1,Integer::sum));
        // 定义小根堆，根据数字频率自小到大排序
        Queue<Integer> pq = new PriorityQueue<>((v1,v2)->counter.get(v1)-counter.get(v2));
        counter.forEach((num,cnt)->{
            if (pq.size()<k){
                pq.offer(num);
            }else if (counter.get(pq.peek())<cnt){
                pq.poll();
                pq.offer(num);
            }
        });
        int[] res = new int[k];
        int idx = 0;
        for (int num:pq){
            res[idx++] = num;
        }
        return res;
    }
}
class Solution103{
    public long maximumTripletValue(int[] nums){
        long ans = 0;
        int maxDiff = 0;
        int preMax = 0;
        for (int x:nums){
            ans = Math.max(ans,(long) maxDiff*x);
            maxDiff = Math.max(maxDiff,preMax-x);
            preMax = Math.max(preMax,x);
        }
        return ans;
    }
}
class Solution63{
    public int maxProfit(int[] prices){
        int cost = Integer.MAX_VALUE,profit = 0;
        for (int price:prices){
            cost = Math.min(cost,price);
            profit = Math.max(profit,price-cost);
        }
        return profit;
    }
}
class Solution64{
    public boolean canJump(int[] nums){
        int mx = 0;
        for (int i=0;mx<nums.length-1;i++){
            if (i>mx) return false;
            mx = Math.max(mx,i+nums[i]);
        }
        return true;
    }
}
class Solution65{
    public int jump(int[] nums){
        int end = 0;
        int maxPosition = 0;
        int steps = 0;
        for (int i =0;i<nums.length-1;i++){
            maxPosition = Math.max(maxPosition,nums[i]+i);
            if (i==end){
                end = maxPosition;
                steps++;
            }
        }
        return steps;
    }
}
class Solution104{
    public TreeNode lcaDeepestLeaves(TreeNode root){
        return dfs(root).getValue();
    }
    private Pair<Integer,TreeNode> dfs(TreeNode node){
        if (node==null){
            return new Pair<>(0,null);
        }
        Pair<Integer,TreeNode> left = dfs(node.left);
        Pair<Integer,TreeNode> right = dfs(node.right);
        if (left.getKey()>right.getKey()){
            return new Pair<>(left.getKey()+1,left.getValue());
        }
        if (left.getKey()<right.getKey()){
            return new Pair<>(right.getKey()+1,right.getValue());
        }
        return new Pair<>(left.getKey()+1,node);
    }
}
class Solution105{
    public int subsetXORSum(int[] nums){
        int or = 0;
        for (int x:nums){
            or |=x;
        }
        return or<<(nums.length-1);
    }
}
class Solution66{
    public List<Integer> partitionLabels(String S){
        char[] s = S.toCharArray();
        int n = s.length;
        int [] last = new int[26];
        for (int i =0;i<n;i++){
            last[s[i]-'a']  = i;
        }
        List<Integer> ans = new ArrayList<>();
         int start = 0,end =0;
        for (int i =0;i<n;i++){
            end = Math.max(end,last[s[i]-'a']);
            if (end == i){
                ans.add(end-start+1);
                start = i+1;
            }
        }
        return ans;
    }
}
class Solution67{
    public int  singleNumber(int[] nums){
        int x=0;
        for (int num:nums){
            x^=num;
        }
        return x;
    }
}
class Solution68{
    public int majorityElement(int[] nums){
        int x=0,votes = 0;
        for (int num:nums){
            if (votes==0) x=num;
            votes +=num ==x?1:-1;
        }
        return x;
    }
}
class Solution106{
    public List<Integer> largestDivisibleSubset(int[] nums){
        Arrays.sort(nums);
        int n  = nums.length;
        int[] f = new int[n];
        Arrays.fill(f,1);
        int k =0;
        for (int i=0;i<n;i++){
            for (int j =0;j<i;j++){
                if (nums[i]%nums[j]==0){
                    f[i] = Math.max(f[i],f[j]+1);
                }
            }
            if (f[k]<f[i]){
                k=i;
            }

        }
        int m =f[k];
        List<Integer> ans = new ArrayList<>();
        for (int i =k;m>0;--i){
            if (nums[k]%nums[i]==0&&f[i]==m){
                ans.add(nums[i]);
                k=i;
                --m;
            }
        }
        return ans;
    }
}
class Solution69{
    public void sortColors(int[] nums) {
        int len = nums.length;
        if (len<2){
            return;
        }
        int zero  = -1;
        int two = len-1;
        int i =0;
        while (i<=two){
            if (nums[i]==0){
                zero++;
                swap(nums,i,zero);
                i++;
            }else if (nums[i]==1){
                i++;
            }else {
                swap(nums,i,two);
                two--;
            }
        }

    }
    private void swap(int[] nums,int index1,int index2){
        int temp = nums[index1];
        nums[index1] = nums[index2];
        nums[index2] = temp;
    }
}
class Solution70{
    private void swap(int[] nums,int i,int j){
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
    private void reverse(int[] nums,int left,int right){
        while (left<right){
            swap(nums,left++,right--);
        }
    }
    public void nextPermutation(int[] nums) {
        int n = nums.length;
        int i = n-2;
        while (i>=0&&nums[i]>=nums[i+1]){
            i--;
        }
        if (i>=0){
            int j = n-1;
            while (nums[j]<=nums[i]){
                j--;
            }
            swap(nums,i,j);
        }
        reverse(nums,i+1,n-1);
    }
}
class Solution71{
    public int findDuolicate(int[] nums){
        int s = 0;
        int f = 0;
        while (true){
            f = nums[f];
            f = nums[f];
            s = nums[s];
            if (s == f) break;
        }
        int ptr = 0;
        while (ptr!=s){
            ptr = nums[ptr];
            s = nums[s];
        }
        return ptr;
    }
}
class Solution72 {
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
class Solution73{
    public List<List<Integer>> generate(int numRows){
        List<List<Integer>> c = new ArrayList<>(numRows);
        if (numRows <= 0) return c;
        for (int i =1;i<numRows;i++){//每一行的实现
            List<Integer> row = new ArrayList<>(i+1);
            row.add(1);//第一个元素
            for (int j =1;j<i;j++){//相加的实现
                row.add(c.get(i-1).get(j-1)+c.get(i-1).get(j));
            }
            row.add(1);//最后一个元素
            c.add(row);
        }
        return c;
    }
}
class Solution107{
    public boolean canPartition(int[] nums){
        int s= 0;
        for (int num:nums){
            s+=num;//总的和
        }
        if (s%2!=0){
            return false;
        }
        s/=2;
        int n = nums.length;
        boolean[][] f = new boolean[n+1][s+1];
        f[0][0] = true;//初始态
        for (int i =0;i<n;i++){
            int x = nums[i];
            for (int j =0;j<=s;j++){
                f[i+1][j] = j>=x&&f[i][j-x]||f[i][j];
            }
        }
        return f[n][s];
    }
}
class Solution74 {
    public int rob(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int N = nums.length;
        int[] dp = new int[N + 1]; // 创建 DP 数组，dp[i] 代表抢劫前 i 间房子时的最大金额

        dp[0] = 0;         // 不抢任何房子
        dp[1] = nums[0];   // 只有一间房子时，抢它

        for (int k = 2; k <= N; k++) {
            dp[k] = Math.max(dp[k - 1], nums[k - 1] + dp[k - 2]);
            // 选择：不抢当前房子(dp[k-1])，或抢当前房子(nums[k-1] + dp[k-2])
        }

        return dp[N]; // 最后一个状态就是最大金额
    }
}
class Solution75{
    public int numSquares(int n){
        int[] f = new int[n+1];
        Arrays.fill(f,Integer.MAX_VALUE);
        f[0]=0;
        for (int i =1;i*i<=n;i++){
            for (int j=i*i;j<=n;j++){
                if (f[j - i * i] != Integer.MAX_VALUE)
                    f[j] = Math.max(f[j],f[j-i*i]+1);
            }
        }
        return f[n];
    }
}
class Solution76{
    public int coinChange(int[] coins,int amount){
        int[] f = new int[amount+1];
        Arrays.fill(f,Integer.MAX_VALUE/2);
        f[0]=0;
        for (int x:coins){
            for (int c = x;c<=amount;c++){
                f[c] = Math.min(f[c],f[c-x]+1);
            }
        }
        int ans = f[amount];
        return ans<Integer.MAX_VALUE/2?ans:-1;
    }
}
class Solution108{
    public int minimumOperations(int[] nums){
        Set<Integer> seen = new HashSet<>();
        for (int i=nums.length-1;i>=0;i--){
            if (!seen.add(nums[i])){//nums[i]在seen中
                return i/3+1;
            }
        }
        return 0;
    }
}
class Solution77{
    public boolean wordBreak(String s,List<String> wordDict){
        int maxlen = 0;
        for (String word:wordDict){
            maxlen = Math.max(maxlen,word.length());
        }
        Set<String> words = new HashSet<>(wordDict);
        int n  = s.length();
        boolean[] f = new boolean[n+1];
        f[0] = true;
        for (int i=1;i<n;i++){
            for (int j =Math.max(i-maxlen,0);j>i;j++){
                if (f[j]&&words.contains(s.substring(j,i))){
                    f[i] = true;
                    break;
                }
            }
        }
        return f[n];
    }
}
class Solution109{
    public int minOperations(int[] nums,int k){
        int min = Arrays.stream(nums).min().getAsInt();//获取最小值
        if (k>min){
            return -1;
        }//不存在
        int distinctCount = (int)Arrays.stream(nums).distinct().count();//记录不同数字个数
        return distinctCount-(k==min?1:0);//等于就1，不等就0
    }
}
class Solution78{
    public int lengthOfLIS(int[] nums){
        if (nums.length==0) return 0;
        int[] dp = new int[nums.length];
        int res = 0;
        Arrays.fill(dp,1);
        for (int i =0;i<nums.length;i++){
            for (int j=0;j<i;j++){
                if (nums[j]<nums[i]) dp[i] = Math.max(dp[i],dp[j]+1);
            }
            res = Math.max(res,dp[i]);
        }
        return res;
    }
}
class Solution79{
    public int maxProduct(int[] nums){
        int max = Integer.MIN_VALUE,imax = 1,imin=1;
        for (int i =0;i<nums.length;i++){
            if (nums[i]<0){
                int tmp = imax;
                imax = imin;
                imin= tmp;
            }
            imax = Math.max(imax*nums[i],nums[i]);
            imin = Math.min(imin*nums[i],nums[i]);
            max = Math.max(max,imax);
        }
        return max;
    }

}
class Solution110{
    private Map<String ,Integer> memo  = new HashMap<>();
    public int numberOfPowerfulInt(int start,int finish,int limit,String s){
        String low = Integer.toString(start-1);
        String high = Integer.toString(finish);
        int n = high.length();
        low = String.format("%0" + n + "d", Integer.parseInt(low));
        int need = n-s.length();

        return dfs(0,true,high,s,need,limit)-dfs(0,true,low,s,need,limit);

    }

    private int dfs(int idx,boolean limitHight,String num,String s,int need,int limit){
        if (idx==num.length()){
            return 1;
        }
        String key = idx + "-" + limitHight + "-" + num.substring(idx);
        if (memo.containsKey(key)){
            return memo.get(key);
        }
        int hi = limitHight?num.charAt(idx)-'0':9;
        int hilimit = Math.min(hi,limit);
        int res = 0;


        if (idx<need){
            for (int d=0;d<=hilimit;d++){
                res+=dfs(idx+1,limitHight&&d==hi,num,s,need,limit);
            }
        }else {
            int x = s.charAt(idx-need)-'0';
            if (x<=hilimit){
                res=dfs(idx+1,limitHight&&x==hi,num,s,need,limit);
            }
        }
        memo.put(key,res);
        return res;
    }

}
class Solution80{
    public int uniquePaths(int m,int n){
        int[][] dp = new int[m][n];
        for (int i =0;i<n;i++) dp[0][i] =1;
        for (int i=0;i<m;i++) dp[i][0] = 1;
        for (int i =1;i<m;i++){
            for (int j=1;j<n;j++){
                dp[i][j] = dp[i-1][j]+dp[i][j-1];
            }
        }
        return dp[m-1][n-1];
    }
}
class Solution81{
    public int minPathSum(int[][] grid){
        for(int i =0;i<grid.length;i++){
            for (int j =0;j<grid[0].length;j++){
                if (i==0&&j==0) continue;
                else if (i==0) grid[i][j] = grid[i][j-1]+grid[i][j];
                else if (j==0) grid[i][j] = grid[i-1][j]+grid[i][j];
                else grid[i][j] += Math.min(grid[i - 1][j], grid[i][j - 1]);
            }
        }
        return grid[grid.length-1][grid[0].length-1];
    }
}
class Solution111{
    public int countSymmetricIntegers(int low,int high){
        int ans = 0;
        for (int x=low;x<=high;++x){
            ans +=f(x);
        }
        return ans;
    }
    private int f(int x){
        String s = ""+x;
        int n  = s.length();
        if (n%2==1){
            return 0;
        }
        int a=0,b=0;
        for (int i=0;i<n/2;++i){
            a +=s.charAt(i)-'0';
        }
        for (int i=n/2;i<n;++i){
            b +=s.charAt(i)-'0';
        }
        return a==b ?1:0;
    }

}
class Solution82 {
    public String longestPalindrome(String s) {
        int n = s.length();
        if (n == 0) return "";

        // 预处理字符串
        char[] t = new char[n * 2 + 3];
        Arrays.fill(t, '#');
        t[0] = '^';
        t[n * 2 + 2] = '$';
        for (int i = 0; i < n; i++) {
            t[i * 2 + 2] = s.charAt(i);
        }

        int[] halfLen = new int[t.length]; // 这里大小应该与 t 一致
        int maxI = 0;
        int boxM = 0, boxR = 0;

        // Manacher's Algorithm
        for (int i = 1; i < t.length - 1; i++) {
            // 计算当前点的初始回文半径
            int hl = (i < boxR) ? Math.min(halfLen[2 * boxM - i], boxR - i) : 1;

            // 尝试扩展回文半径
            while (t[i - hl] == t[i + hl]) {
                hl++;
            }

            halfLen[i] = hl;

            // 更新右边界
            if (i + hl > boxR) {
                boxM = i;
                boxR = i + hl;
            }

            // 更新最长回文中心
            if (halfLen[i] > halfLen[maxI]) {
                maxI = i;
            }
        }

        // 计算原字符串中的起始索引
        int start = (maxI - halfLen[maxI]) / 2;
        int end = start + halfLen[maxI] - 1;

        return s.substring(start, end);
    }
}
class Solution83{
    public int longestCommonSubsequence(String text1,String text2){
        char[] s = text1.toCharArray();
        char[] t = text2.toCharArray();
        int n = s.length;
        int m = t.length;
        int[][] f = new int[n+1][m+1];
        for (int i =0;i<n;i++){
            for (int j =0;j<m;j++){
                f[i+1][j+1] = s[i]==t[j]?f[i][j]+1:
                        Math.max(f[i][j+1],f[i+1][j]);
            }
        }
        return f[n][m];
    }
}
class Solution84{
    public int minDistance(String text1,String text2){
        char[] t = text2.toCharArray();
        int m = t.length;
        int[] f = new int[m+1];
        for (int j=1;j<=m;j++){
            f[j] = j;
        }
        for (char x:text1.toCharArray()){
            int pre = f[0];
            f[0]++;
            for (int j =0;j<m;j++){
                int tmp = f[j+1];
                f[j+1] = x==t[j]?pre:Math.min(Math.min(f[j + 1], f[j]), pre) + 1;
                pre = tmp;
            }
        }
        return f[m];
    }

}
class Solution112{
    public long countGoodIntegers(int n,int k){
        int[] factorial = new int[n+1];
        factorial[0]=1;
        for (int i =1;i<=n;i++){
            factorial[i]=factorial[i-1]*i;
        }//阶乘
        long ans = 0;
        Set<String> vis = new HashSet<>();//去重
        int base = (int)Math.pow(10,(n-1)/2);//前半部分的起始值
        for (int i=base;i<base*10;i++){
            String s = Integer.toString(i);
            s+=new StringBuilder(s).reverse().substring(n%2);//构造回文
            if (Long.parseLong(s)%k>0){
                continue;
            }
            char[] sortedS = s.toCharArray();
            Arrays.sort(sortedS);
            if (!vis.add(new String(sortedS))){
                continue;
            }//去重
            int[] cnt = new int[10];//次数
             for (char c:sortedS){
                 cnt[c-'0']++;
            }
             int res = (n-cnt[0])*factorial[n-1];//不能以0为开头，然后乘以(n-1)!
             for (int c:cnt){
                 res/=factorial[c];
             }//去掉重复数字
             ans+=res;
        }
        return ans;
    }
}
class Solution113{
    private static final int MOD = 1_000_000_007;
    public int countGoodNumbers(long n){
        return (int)(pow(5,(n+1)/2)*pow(4,n/2)%MOD);
    }
    private long pow(long x,long n){
        long res = 1;
        while (n>0){
            if ((n&1)>0){
                res = res*x%MOD;
            }
            x =x*x%MOD;
            n>>=1;
        }
        return res;
    }

}
class Solution85{
    public int trap(int[] height){
        int ans  = 0;
        int left = 0;
        int right  = height.length-1;
        int preMax = 0;
        int sufMax=0;
        while (left<right){
            preMax = Math.max(preMax,height[left]);
            sufMax  = Math.max(sufMax,height[right]);
            ans +=preMax<sufMax?preMax-height[left++]:sufMax-height[right--];
        }
        return ans;
    }
}
class Solution114{
    public int countGoodTriplets(int[] arr,int a,int b,int c){
        int n = arr.length;
        int ans = 0;
        for (int i = 0;i<n;i++){
            for (int j=i+1;j<n;j++){
                for (int k=j+1;k<n;k++){
                    if (Math.abs(arr[i]-arr[j])<=a&&Math.abs(arr[j]-arr[k])<=b&&Math.abs(arr[i]-arr[k])<=c){
                        ans++;
                    }
                }
            }
        }
        return ans;
    }
}
class Solution86{
    public int[] maxSlidingWindow(int[] nums, int k ){
        if (nums.length==0||k==0) return new int[0];
        Deque<Integer> deque = new LinkedList<>();
        int[] res = new int[nums.length-k+1];
        for (int j =0,i=1-k;j<nums.length;i++,j++){
            if (i>0&&deque.peekFirst()==nums[i-1])
                deque.removeFirst();
            while (!deque.isEmpty()&&deque.peekLast()<nums[j])
                deque.removeLast();
            deque.addLast(nums[j]);
            if (i>=0)
                res[i] =deque.peekFirst();

        }
        return res;
    }

}
class Solution115{
    int n;
    int[] tree = new int [10010];
    public long goodTriplets(int[] nums1,int[] nums2){
        n = nums1.length;
        long ans =0;
        Map<Integer,Integer> num2idx = new HashMap<>();
        for (int i=0;i<n;i++){
            num2idx.put(nums1[i],i);
        }
        for (int i=0;i<n;i++){
            nums2[i] = num2idx.get(nums2[i]);
        }
        for (int i =0;i<n;i++){
            int l = query(nums2[i]+1);
            int t = i-l;
            int r = (n - nums2[i] - 1) - t;
            add(nums2[i]+1,1);
            ans+=1L*l*r;
        }
        return ans;
    }
    int lowbit(int x){
        return x&-x;
    }
    int query(int x){
        int ans = 0;
        for (int i=x;i>0;i-=lowbit(i)) ans+=tree[i];
        return ans;
    }
    void add(int x,int u){
        for (int i=x;i<=n;i+=lowbit(i)) tree[i]+=u;
    }

}
class Solution116{
    public long countGood(int[] nums,int k){
        long ans= 0;
        Map<Integer,Integer> cnt = new HashMap<>();
        int pairs= 0;
        int left = 0;
        for (int x:nums){
            int c = cnt.getOrDefault(x,0);
            pairs+=c;
            cnt.put(x,c+1);
            while (pairs>=k){//至少k对
                x=nums[left];
                c= cnt.get(x);
                pairs -=(c-1);
                cnt.put(x,c-1);
                left++;//右移动
            }
            ans+=left;
        }
        return ans;
    }
}
class Solution117{
    public int countPairs(int[] nums,int k){
        int ans =0;
        for (int j=1;j<nums.length;++j){
            for(int i=0;i<j;++i){
                ans +=nums[i]==nums[j]&&(i*j%k)==0?1:0;
            }
        }
        return ans;
    }
}
class Solution87{
    public int firstMissingPositive(int[] nums){
        int n = nums.length;
        Set<Integer> set  =new HashSet<>();
        int min = Integer.MAX_VALUE;
        for (int num:nums){
            if (num<min){
                min = num;
            }
            set.add(num);

        }//找到最小值
        if (min>1&&!set.contains(1)) return 1;
        while (set.contains(min)){
            min++;
            if (min<0) min = 1;
            if (set.contains(min)) min++;
            else {
                if (min<=0) min++;
            }//将负数Min++直到为1
        }
        return min>0?min:1;//不在集合中，大于0直接返回
    }
}
class Solution118{
    public long countBadPairs(int[] nums){
        int n = nums.length;
        long ans  = (long)n*(n-1)/2;
        Map<Integer,Integer> cnt = new HashMap<>();
        for (int i =0;i<n;i++){
            int x = nums[i]-i;
            int c = cnt.getOrDefault(x,0);
            ans -=c;
            cnt.put(x,c+1);
        }
        return ans;
    }
}
class Solution118a{
    public int numIdenticalPairs(int[] nums){
        int n = nums.length;
        int ans = 0;
        Map<Integer,Integer> cnt = new HashMap<>();
        for (int x:nums){
            int c = cnt.getOrDefault(x,0);
            ans+=c;
            cnt.put(x,c+1);
        }
        return ans;
    }
}
class Solution119{
    public long countFairPairs(int[] nums,int lower, int upper){
        Arrays.sort(nums);
        long ans = 0;
        for (int j=0;j<nums.length;j++){
            int r = lowerBound(nums,j,upper-nums[j]+1);
            int l = lowerBound(nums,j,lower-nums[j]);
            ans += r-l;
        }
        return ans;
    }
    private int lowerBound(int[] nums,int right,int target){
        int left=-1;
        while (left+1<right){
            int mid = (left+right)>>>1;
            if (nums[mid]>=target){
                right=mid;
            }else {
                left=mid;
            }
        }
        return right;
    }
}
class Solution704{
    public int search(int[] nums, int target) {
        int left=0,right = nums.length-1;
        while (left<=right){
            int mid = left+(right-left)/2;
            if (nums[mid]==target){
                return mid;
            }
            else if (nums[mid]<target){
                left=mid+1;
            }
            else {
                right = mid -1;
            }
        }
        return -1;
    }
}
class Solution120{
    public int numRabbits(int[] answers){
        int ans= 0;
        Map<Integer,Integer> left = new HashMap<>();
        for (int x:answers){
            int c = left.getOrDefault(x,0);
            if (c==0){
                ans+=x+1;
                left.put(x,x);
            }else {
                left.put(x,c-1);
            }
        }
        return ans;
    }
}
class Solution744{
    public char nextGreatestLetter(char[] letters, char target){
        int l =0,r=letters.length-1;
        while (l<r){
            int mid=(l+r)>>1;
            if (letters[mid]>target) r = mid;
            else l=mid+1;
        }
        return letters[l]>target?letters[l]:letters[0];
    }
}
class Solution2529 {
    public int maximumCount(int[] nums) {
        int n = nums.length;
        int negativeCount = findFirstIndex(nums, 0); // 第一个 >= 0 的位置就是负数个数
        int positiveCount = n - findFirstIndex(nums, 1); // 第一个 > 0 的位置就是正数个数
        return Math.max(negativeCount, positiveCount);
    }

    // 返回第一个 >= target 的索引
    private int findFirstIndex(int[] nums, int target) {
        int l = 0, r = nums.length;
        while (l < r) {
            int mid = l + ((r - l) >> 1); // 防止溢出
            if (nums[mid] < target) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        return l;
    }
}







