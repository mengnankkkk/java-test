public class Solution10 {
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1; // nums1 中最后一个有效元素的索引
        int j = n - 1; // nums2 中最后一个有效元素的索引
        int index = m + n - 1; // 合并后数组的最后一个位置的索引

        // 从后向前合并两个数组
        while (i >= 0 && j >= 0) {
            if (nums1[i] <= nums2[j]) {
                nums1[index--] = nums2[j--];
            } else {
                nums1[index--] = nums1[i--];
            }
        }

        // 如果 nums2 中还有剩余元素，继续填入 nums1 中
        while (j >= 0) {
            nums1[index--] = nums2[j--];
        }

        // 如果 nums1 中有剩余元素，不需要额外处理，因为它们已经在正确的位置上
    }
}