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
class Solution2845{
    public long countInterestingSubarrays8(List<Integer> nums, int modulo, int k){
        int n  = nums.size();
        int[] prefix = new int[n+1];
        for (int i =1;i<=n;i++){
            prefix[i] = prefix[i-1]+(nums.get(i-1)%modulo==k?1:0);

        }
        long count = 0;
        for (int l =0;l<n;l++){
            for (int r = l;r<n;r++){
                int cnt = prefix[r+1] -prefix[l];
                if (cnt%modulo==k){
                    count++;
                }
            }
        }
        return count;
    }
    public long countInterestingSubarrays(List<Integer> nums, int modulo, int k){
        int n  = nums.size();
        int[] prefix = new int[n+1];
        for (int i =1;i<=n;i++){
            prefix[i] = prefix[i-1]+(nums.get(i-1)%modulo==k?1:0);
        }
        Map<Integer,Integer> countMap = new HashMap<>();
        countMap.put(0,1);
        long ans = 0;
        for (int i =1;i<=n;i++){
            int currentMod = prefix[i]%modulo;
            int target = (currentMod-k+modulo)%modulo;
            ans += countMap.getOrDefault(target,0);
            countMap.put(currentMod,countMap.getOrDefault(currentMod,0)+1);//更新
        }
        return ans;

    }
}
class Solution3488{
    public List<Integer> solveQueries(int[] nums, int[] queries){
        Map<Integer,List<Integer>> indices = new HashMap<>();
        for (int i =0;i<nums.length;i++){
            indices.computeIfAbsent(nums[i],k->new ArrayList<>()).add(i);
        }//构建hash表，indices存储

        int n = nums.length;
        for (List<Integer> p :indices.values()){
            int i0 = p.get(0);
            p.add(0,p.get(p.size()-1)-n);//循环向左的哨兵
            p.add(i0+n);//循环向右的哨兵
        }
        List<Integer> ans  = new ArrayList<>(queries.length);
        for (int i :queries){
            List<Integer> p = indices.get(nums[i]);
            if (p.size()==3){
                ans.add(-1);//没有，只有一次
            }else {
                int j = Collections.binarySearch(p,i);//二分查找位置，i在p的位置
                ans.add(Math.min(i-p.get(j-1),p.get(j+1)-i));//比较前一个和后一个
            }
        }
        return ans;
    }
}
class Solution2444{
    public long countSubarrays(int[] nums, int minK, int maxK){
        long ans = 0;
        int minI = -1,maxI = -1,i0=-1;
        for (int i =0;i<nums.length;i++){
            int x = nums[i];
            if (x==minK){
                minI = i;
            }
            if (x==maxK){
                maxI = i;
            }
            if (x<minK||x>maxK){
                i0=i;//i0不包含在里面
            }
            ans +=Math.max(Math.min(minI,maxI)-i0,0);
        }
        return ans;
    }
}
class Solution2563{
    public long countFairPairs(int[] nums,int lower, int upper){
        Arrays.sort(nums);
        long ans = 0;
        for (int j=0;j<nums.length;j++){
            int r = lowerBound(nums,j,upper-nums[j]+1);
            int l = lowerBound(nums,j,lower-nums[j]);
            ans += r-l;
        }
        return ans;
    }
    private int lowerBound(int[] nums,int right,int target){
        int left=-1;
        while (left+1<right){
            int mid = (left+right)>>>1;
            if (nums[mid]>=target){
                right=mid;
            }else {
                left=mid;
            }
        }
        return right;
    }
}