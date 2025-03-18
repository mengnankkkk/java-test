import com.oracle.xmlns.internal.webservices.jaxws_databinding.XmlWebEndpoint;

import java.security.PublicKey;
import java.time.temporal.Temporal;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

class ListNode {
    int val;
    ListNode next;
    ListNode(int x) { val = x; next = null; }
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
                ListNode pre = new ListNode(0);
                ListNode cur = pre;
                int carry = 0;
                while (l1 != null || l2 != null) {
                    int x = l1 == null ? 0 : l1.val;
                    int y = l2 == null ? 0 : l2.val;
                    int sum = x + y + carry;

                    carry = sum / 10;
                    sum = sum % 10;
                    cur.next = new ListNode(sum);

                    cur = cur.next;
                    if (l1 != null) {
                        l1 = l1.next;
                    }
                    if (l2 != null) {
                        l2 = l2.next;
                    }
                }
                if (carry == 1) {
                    cur.next = new ListNode(carry);
                }
                return pre.next;
            }
        }
        class Solution14 {
            public ListNode removeNthFromEnd(ListNode head, int n) {
                ListNode pre = new ListNode(0);
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
        ListNode pre = new ListNode(0);
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
        ListNode listNode = new ListNode(arr[0]);
        cur = listNode;
        for (int i=1;i<arr.length;i++){
            cur.next = new ListNode(arr[i]);
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