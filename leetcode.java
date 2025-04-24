import java.lang.reflect.Array;
import java.util.*;

public class leetcode {
}
class Solution2300{
    public int[] successfulPairs(int[] spells, int[] potions, long success){
        Arrays.sort(potions);
        for (int i=0;i<spells.length;i++){
            long target = (success-1)/spells[i];
            if (target<potions[potions.length-1]){
                spells[i] = potions.length-upperBound(potions,(int)target);
            }else {
                spells[i]=0;

            }
        }
        return spells;
    }
    private int upperBound(int[] nums,int target){
        int left = -1,right = nums.length;
        while (left+1<right){
            int mid = (left+right)>>>1;
            if (nums[mid]>target){
                right = mid;
            }
            else {
                left = mid;
            }
        }
        return right;
    }
}
class Solution2145{
    public int numberOfArrays(int[] differences,int lower,int upper){
        long s = 0,minS = 0,maxS = 0;
        for (int d:differences){
            s+=d;
            minS = Math.min(minS,s);
            maxS = Math.max(maxS,s);
        }
        return (int) Math.max(upper-lower-maxS+minS+1,0);
    }
}
class Solution1385{
    public int findTheDistanceValue(int[] arr1, int[] arr2, int d){
        Arrays.sort(arr2);
        int ans = 0;
        for (int x:arr1){
            int i  = Arrays.binarySearch(arr2,x-d);
            if (i<0){
                i=~i;
            }
            if (i==arr2.length||arr2[i]>x+d){
                ans++;
            }
        }
        return ans;
    }
}
class Solution2389{
    private int upperBound(int[] nums,int target){
        int left = -1,right = nums.length;
        while (left+1<right){
            int mid = left+(right-left)/2;
            if (nums[mid]>target){
                right = mid;
            }
            else {
                left = mid;
            }
        }
        return right;
    }
    public int[] answerQueries(int[] nums, int[] queries){
        Arrays.sort(nums);
        for (int i =1;i<nums.length;i++){
            nums[i]+=nums[i-1];
        }
        for (int i =0;i<queries.length;i++){
            queries[i] = upperBound(nums,queries[i]);
        }
        return queries;
    }
}
class Solution1170{
    public int[] numSmallerByFrequency(String[] queries, String[] words){
        int n = words.length;
        int[] nums = new int[n];
        for (int i =0;i<n;++i){
            nums[i] = f(words[i]);
        }
        Arrays.sort(nums);
        int m  = queries.length;
        int[] ans = new int[m];
        for (int i =0;i<m;++i){
            int x = f(queries[i]);
            int l=0,r= n;
            while (l<r){
                int mid = (l+r)>>1;
                if(nums[mid]>x){
                    r = mid;
                }
                else {
                    l = mid+1;
                }
            }
            ans[i]=n-l;
        }
        return ans;

    }
    private int f(String s){
        int[] cnt = new int[26];
        for (int i =0;i<s.length();++i){
            ++cnt[s.charAt(i)-'a'];
        }
        for (int x:cnt){
            if (x>0){
                return x;
            }
        }
        return 0;
    }
}
class Solution1399{
    private int calcDigitSum(int num){
        int ds = 0;
        while (num>0){
            ds +=num%10;
            num/=10;
        }
        return ds;
    }
    public int countLargestGroup(int n){
        int m  = String.valueOf(n).length();
        int[] cnt = new int[m*9+1];
        int maxCnt = 0;
        int ans = 0;
        for (int i =1;i<=n;i++){
            int ds = calcDigitSum(i);
            cnt[ds]++;
            if (cnt[ds]>maxCnt){
                maxCnt = cnt[ds];
                ans = 1;
            }else if (cnt[ds]==maxCnt){
                ans++;
            }
        }
        return ans;

    }
}
class Solution2799{
    public int countCompleteSubarraysA(int[] nums){
        Set<Integer> set = new HashSet<>();
        for (int x:nums){
            set.add(x);
        }
        int k  = set.size();
        Map<Integer,Integer> cnt =  new HashMap<>();
        int ans = 0;
        int left = 0;
        for (int x:nums){
            cnt.merge(x,1,Integer::sum);
                while (cnt.size()==k){
                    int out = nums[left];
                    if (cnt.merge(out,-1,Integer::sum)==0){
                        cnt.remove(out);
                    }
                    left++;
                }
                ans +=left;
        }
        return ans;
    }
    public int countCompleteSubarraysB(int[] nums){
        Set<Integer> set = new HashSet<>();
        for (int x:nums){
            set.add(x);
        }
        int res = 0;
        int left = 0;
        Map<Integer,Integer> map = new HashMap<>();
        int count = 0;
        for (int right = 0;right<nums.length;right++){
            map.put(nums[right],map.getOrDefault(nums[right],0)+1);
            while (map.keySet().size()==set.size()){
                res+=nums.length-right;
                map.put(nums[left],map.get(nums[left]-1));
                if (map.get(nums[left])==0) map.remove(nums[left]);
                left++;
            }
        }
        return res;
    }
}
class RangeFreQuery{
    private int lowBound(List<Integer> a,int target){
        int left = -1,right = a.size();
        while (left+1<right){
            int mid = (left+right)>>>1;
            if (a.get(mid)<target){
                left = mid;
            }else {
                right = mid;
            }
        }
        return right;
    }
    private final Map<Integer, List<Integer>> pos = new HashMap<>();
    public RangeFreQuery(int[] arr){
        for (int i=0;i<arr.length;i++){
            pos.computeIfAbsent(arr[i],k->new ArrayList<>()).add(i);
        }

    }
    public int query(int left,int right,int value){
        List<Integer> a = pos.get(value);
        if (a==null){
            return 0;
        }
        return lowBound(a,right+1)-lowBound(a,left);
    }

}