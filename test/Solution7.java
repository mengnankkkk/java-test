public class Solution7 {
    int s;
    public int mySqrt(int x){
        s=x;
        if(x==0) return 0;
        return ((int)(sqrts(x)));//主体过测试的主方法，返回平方
    }
    public double sqrts (double x){//设定方法
        double res = (x+s/x)/2;
        if(res == x){
            return x;
        }
        else{
            return sqrts(res);//一直重复这个过程直到x==res的时候
        }
    }
}
