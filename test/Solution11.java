public class Solution11 {
    public boolean isPalindrome(String s) {
        int l = 0, r = s.length() - 1;//前后项

        while (l < r) {
            // 跳过非字母和非数字字符
            while (l < r && !Character.isLetterOrDigit(s.charAt(l))) {
                l++;
            }
            while (l < r && !Character.isLetterOrDigit(s.charAt(r))) {
                r--;
            }

            // 比较字符
            if (l < r) {
                if (Character.toLowerCase(s.charAt(l)) != Character.toLowerCase(s.charAt(r))) {
                    return false;
                }
                l++;
                r--;
            }
        }
        return true;
    }
}
