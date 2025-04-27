// 定义链表节点类
class ListNode {
    int val;
    ListNode next;
    ListNode(int x) { val = x; }
}

// 解决方案类
public class Solution9 {
    public ListNode deleteDuplicates(ListNode head) {
        ListNode cur = head;
        while (cur != null && cur.next != null) {
            if (cur.val == cur.next.val) {
                cur.next = cur.next.next;
            } else {
                cur = cur.next;
            }
        }
        return head;
    }

    // 主方法，用于测试
    public static void main(String[] args) {
        // 创建测试链表：1 -> 1 -> 2 -> 3 -> 3
        ListNode head = new ListNode(1);
        head.next = new ListNode(1);
        head.next.next = new ListNode(2);
        head.next.next.next = new ListNode(3);
        head.next.next.next.next = new ListNode(3);

        Solution9 solution = new Solution9();
        ListNode result = solution.deleteDuplicates(head);

        // 输出处理后的链表
        while (result != null) {
            System.out.print(result.val + " ");
            result = result.next;
        }
        // 期望输出：1 2 3
    }
}
