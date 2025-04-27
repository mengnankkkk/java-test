public class Main {
    public static void main(String[] args) {
        // 示例输入
        int[] nums1 = {1, 2, 3, 0, 0, 0}; // nums1 数组，大小为 m + n
        int m = 3; // nums1 中的有效元素个数
        int[] nums2 = {2, 5, 6}; // nums2 数组
        int n = 3; // nums2 中的有效元素个数

        // 创建 Solution10 实例
        Solution10 solution10 = new Solution10();

        // 调用 merge 方法
        solution10.merge(nums1, m, nums2, n);

        // 输出合并后的 nums1 数组
        for (int num : nums1) {
            System.out.print(num + " ");
        }
    }
}
