import java.nio.file.LinkOption;
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
class Solution7{
    public int lenOfLongesSubstring(String s){
        Map<Character,Integer> dic  = new HashMap<>();
        int i=-1,res = 0,len =s.length();
        for(int j=0;i<len;j++){
            if (dic.containsKey(s.charAt(j)))
                i = Math.max(i,dic.get(s.charAt(j)));
            dic.put(s.charAt(j),j);
            res = Math.max(res,j-i);
        }
        return res;
    }
}
class Solution8 {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;

        ListNode A = headA, B = headB;

        while (A != B) {
            A = (A != null) ? A.next : headB;
            B = (B != null) ? B.next : headA;
        }

        return A;
    }
}

class ListNode {
    int val;
    ListNode next;

    ListNode(int x) {
        val = x;
        next = null;
    }
}
class Solution9 {
    public ListNode reverList(ListNode head) {
        ListNode cur = head, pre = null;
        while (cur != null) {
            ListNode tmp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = tmp;

        }
        return pre;
    }

    public boolean isPalindrome(ListNode head) {
        ListNode mid = middleNode(head);
        ListNode head2 = reverList(mid);
        while (head2 != null) {
            if (head.val != head2.val) {
                return false;
            }
            head = head.next;
            head2 = head2.next;
        }
        return true;
    }

    private ListNode middleNode(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }

    class Solution10 {
        public boolean hasCycle(ListNode head) {
            ListNode slow = head, fast = head;
            while (fast != null && fast.next != null) {
                slow = slow.next;
                fast = fast.next.next;
                if (fast == slow) {
                    return true;
                }
            }
            return false;
        }
    }

    class Solution11 {
        public ListNode detectCycle(ListNode head) {
            ListNode fast = head, slwo = head;
            while (fast != null && fast.next != null) {
                fast = fast.next.next;
                slwo = slwo.next;
                if (fast == slwo) {
                    fast = head;
                    while (slwo != fast) {
                        slwo = slwo.next;
                        fast = fast.next;
                    }
                    return fast;
                }
            }
            return null;
        }
    }

    class Solution12 {
        public ListNode mergeTowLists(ListNode l1, ListNode l2) {
            if (l1 == null) {
                return l2;
            } else if (l2 == null) {
                return l1;
            } else if (l1.val < l2.val) {
                l1.next = mergeTowLists(l1.next, l2);
                return l1;
            } else {
                l2.next = mergeTowLists(l1, l2.next);
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
}