public class stcmp {
    public static int strcmp(String str1,String str2){
        int index1 = 0,index2 = 0;

        while (index1<str1.length()&&index2<str2.length())
            if (str1.charAt(index1)==str2.charAt(index2)){
                index1++;
                index2++;
            }else if (str1.charAt(index1)<str2.charAt(index2)){
                return -1;
            }else {
                return 1;
            }
        if (str1.length()<str2.length()){
            return -1;
        }else if (str1.length()>str2.length()){
            return 1;
        }
        else {
            return 0;
        }
    }
}


