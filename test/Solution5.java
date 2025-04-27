
public class Solution5 {
    public int[] plusOne(int[] digits){
        for(int i = digits.length-1;i>=0;i--){//从最后一个开始遍历
            digits[i]++;
            digits[i] = digits[i] % 10;
            if(digits[i] != 0)return digits;
        }
        digits = new int[digits.length + 1];//如果遍历完了，还需要进位的话，长度加1，第一位变成1
        digits[0] = 1;
        return digits;
    }
}
