class Solution6{
    public String addBinary(String a,String b){
        StringBuilder ans = new StringBuilder();
        int ca = 0;//初始化
        for(int i = a.length() - 1,j=b.length() - 1;i>=0||j>=0;i--,j--){//从最后一位开始往前递归
            int sum = ca;
            sum += i>=0 ? a.charAt(i) - '0' : 0;//往前一直a.charAt(i) - '0' 将字符 '0' 或 '1' 转换为整数 0 或 1。
            sum += j>=0 ? b.charAt(j) - '0' : 0;
            ans.append(sum % 2);
            ca = sum /2;//更新下一位
        }
        ans.append(ca == 1?ca : "");
        return ans.reverse().toString();
    }
}