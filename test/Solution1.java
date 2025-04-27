public class Solution1 {
    // 方法：找到字符串数组中最长的公共前缀
    public String longestCommonPrefix(String[] strs) {
        // 如果输入数组为空，直接返回空字符串
        if (strs.length == 0)
            return ""; // 没有字符的空情况，其字符的长度为0

        // 初始化前缀字符串为第一个字符串
        String ans = strs[0];

        // 从第二个字符串开始遍历数组
        for (int i = 1; i < strs.length; i++) {
            int j = 0;
            // 在当前字符串和前缀字符串中逐字符比较，直到遇到不同字符或者到达其中一个字符串的末尾
            for (; j < ans.length() && j < strs[i].length(); j++) {
                if (ans.charAt(j) != strs[i].charAt(j))
                    break;
            }
            // 更新前缀字符串为当前比较结果的前缀部分
            ans = ans.substring(0, j);
            // 如果前缀字符串为空，直接返回空字符串
            if (ans.equals(""))
                return ans;
        }
        // 返回最终的最长公共前缀
        return ans;
    }

    // 主方法，用于测试和验证
    public static void main(String[] args) {
        Solution1 solution = new Solution1();
        // 示例字符串数组
        String[] strs = {"flower", "flow", "flight"};
        // 输出最长公共前缀
        System.out.println("Longest common prefix: " + solution.longestCommonPrefix(strs));
    }
}
