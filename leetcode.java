import com.sun.jmx.snmp.SnmpNull;
import com.sun.scenario.effect.Brightpass;
import javafx.util.Pair;

import java.rmi.MarshalException;
import java.util.*;
import java.util.jar.JarEntry;


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
    }//开区间的二分查找

}
class Solution2070{
    private int upperBound(int[][] items, int target){
        int left = -1;
        int right = items.length;
        while (left+1<right){
            int mid = (left+right)>>>1;
            if (items[mid][0]>target){
                right =mid;
            }
            else{
                left = mid;
            }
        }
        return right;
    }
    public int[] maximumBeauty(int[][] items, int[] queries){
        Arrays.sort(items,(a,b)->a[0]-b[0]);//排序规则
        for (int i =1;i<items.length;i++){
            items[i][1] = Math.max(items[i][1],items[i-1][1]);//更新美丽值，是前一个位置的最大美丽值
        }
        for (int i =0;i<queries.length;i++){
            int j = upperBound(items,queries[i]);
            queries[i] =j>0?items[j-1][1]:0;//查找到了，就是前一个的最大美丽值，不是的话就是0
        }
        return queries;
    }
}
class Solution3392{
    public int countSubarrays(int[] nums){
        int ans = 0;
        for (int i =2;i<nums.length;i++){
            if ((nums[i-2]+nums[i])*2==nums[i-1]){
                ans++;
            }
        }
        return ans;
    }
}
class SnapshotArray{
    private int curSnapId;
    private final Map<Integer,List<int[]>> history = new HashMap<>();

    public SnapshotArray(int length){

    }
    public void set(int index,int val){
        history.computeIfAbsent(index,k->new ArrayList<>()).add(new int[]{curSnapId,val});
    }
    public int snap(){
        return curSnapId++;
    }
    public int get(int index,int snapId){
        if (!history.containsKey(index)){
            return 0;
        }
        List<int[]> h = history.get(index);
        int j  = search(h,snapId);
        return j<0?0:h.get(j)[1];
    }
    private int search(List<int[]> h,int x){
        int left = -1;
        int right = h.size();
        while (left+1<right){
            int mid = (left+right)>>>1;
            if (h.get(mid)[0]<=x){
                left = mid;
            }else {
                right = mid;
            }
        }
        return left;
    }
}
class TimeMap {
    private final Map<String, List<Info>> tmap;

    static class Info {
        String value;
        int timestamp;

        public Info(String value, int timestamp){
            this.value = value;
            this.timestamp = timestamp;
        }
    }

    public TimeMap() {
        tmap = new HashMap<>();
    }

    public void set(String key, String value, int timestamp) {
        tmap.computeIfAbsent(key,k -> new ArrayList<>()).add(new Info(value, timestamp));
    }

    public String get(String key, int timestamp) {
        if (!tmap.containsKey(key)){
            return "";
        }
        List<Info> tmp = tmap.get(key);
        int left = -1, right = tmp.size();
        while (left + 1 < right){
            int mid = (left + right) >>> 1;
            if (tmp.get(mid).timestamp > timestamp){
                right = mid;
            } else {
                left = mid;
            }
        }
        return left < 0 ? "" : tmp.get(left).value;
    }
}
class Solution2302{
    public long countSubarrays(int[] nums,long k){
        long ans = 0;
        long sum = 0;
        int left = 0;
        for (int right = 0;right<nums.length;right++){
            sum +=nums[right];
            while (sum*(right-left+1)>=k){
                sum -=nums[left];
                left++;//下一项
            }
            ans +=right-left+1;
        }
        return ans;
    }
}
class Solution658{
    public List<Integer> findClosestElements(int[] arr, int k, int x){
        List<Integer> list = new ArrayList<>();//存放数据
        int n = arr.length;
        int left = 0;
        int right  = n-k;
        while (left<=right){
            int mid = (left+right)>>>1;
            if (mid+k<n&&x-arr[mid]>arr[mid+k]-x){
                left= mid+1;
            }else {
                right = mid-1;
            }
        }
        for (int i =left;i<left+k;i++){//找到<=x
            list.add(arr[i]);
        }
        return list;
    }
}
class Solution1287{
    public int findSpecialInteger(int[] arr){
        int n  = arr.length;
        int l  = 0,r = n/4;
        while (r<n){
            if (arr[l]==arr[r]) return arr[r];
            l++;
            r++;
        }
        return -1;
    }
}
class Solution2962{
    public long countSubarrays(int[] nums, int k){
        int mx =0;
        for (int x:nums){
            mx = Math.max(mx,x);//找到最大值
        }
        long ans = 0;
        int cntMx = 0,left = 0;
        for (int x:nums){
            if (x==mx){
                cntMx++;
            }
            while (cntMx==k){
                if (nums[left]==mx){
                    cntMx--;//窗口出去的话
                }
                left++;//移动
            }
            ans +=left;//数量为left
        }
        return ans;
    }
}
class Solution2071{
    public int maxTaskAssign(int[] tasks, int[] workers, int pills, int strength){
        Arrays.sort(tasks);
        Arrays.sort(workers);

        int left  = 0;
        int right = Math.min(tasks.length,workers.length)+1;
        while (left+1<right){
            int mid = (left+right)>>>1;
            if (check(tasks,workers,pills,strength,mid)){
                left = mid;
            }else {
                right = mid;
            }

        }
        return left;

    }
    private boolean check(int[] tasks, int[] workers, int pills, int strength, int k){
        Deque<Integer> validTasks = new ArrayDeque<>();
        int i =0;
        for (int j =workers.length-k;j<workers.length;j++){
            int w = workers[j];
            while (i<k&&tasks[i]<=w+strength){
                validTasks.add(tasks[i]);
                i++;

            }
            if (validTasks.isEmpty()){
                return false;
            }
            if(w>=validTasks.peekFirst()){
                validTasks.pollFirst();
            }else{
                if (pills==0) return false;
                pills--;
                validTasks.pollLast();
            }

        }
        return true;
    }
}
class Solution1456{
    public int maxVowels(String S,int k ){
        char[] s = S.toCharArray();
        int ans  = 0;
        int vowel  = 0;

        for (int i =0;i<s.length;i++){
            if (s[i] == 'a' || s[i] == 'e' || s[i] == 'i' || s[i] == 'o' || s[i] == 'u'){
                vowel++;
            }
            if (i<k-1){//窗口大小不到k
                continue;
            }
            ans = Math.max(ans,vowel);
            char out = s[i-k+1];
            if (out == 'a' || out == 'e' || out == 'i' || out == 'o' || out == 'u'){
                vowel--;
            }
        }
        return ans;

    }
}
class Solution643{
    public double findMaxAverage(int[] nums, int k){
        int maxS = Integer.MIN_VALUE;
        int s=  0;
        for (int i =0;i<nums.length;i++){
            s +=nums[i];
            if (i<k-1){
                continue;
            }
            maxS = Math.max(maxS,s);
            s -=nums[i-k+1];
        }
        return (double) maxS/k;
    }
}
class Solution1343{
    public int numOfSubarrays(int[] arr, int k, int threshold){
        int ans = 0;
        int s=  0;
        for (int i =0;i<arr.length;i++){
            s +=arr[i];
            if (i<k-1){
                continue;
            }
            if (s>=threshold*k){
                ans++;
            }
            s -=arr[i-k+1];
        }
        return ans;
    }
}
class Solution838{
    public String pushDominoes(String dominoes){
        char[] s = ("L"+dominoes+"R").toCharArray();
        int pre = 0;
        for (int i =1;i<s.length;i++){
            if (s[i]=='.'){
                continue;
            }//没推动
            if (s[i]==s[pre]){
                Arrays.fill(s,pre+1,i,s[i]);//从pre+1到i都变成s[i]
            }else if (s[i]=='L'){
                Arrays.fill(s,pre+1,(pre+i+1)/2,'R');//把前一半的点全部变成 R，后一半的点全部变成 L。
                Arrays.fill(s,(pre+i)/2+1,i,'L');

            }
            pre = i;
        }
        return new String(s,1,s.length-2);//返回的时候去掉L与R
    }
}
class Solution2090{
    public int[] getAveragesA(int[] nums, int k){
        int n = nums.length;
        int[] avgs = new int [n];
        Arrays.fill(avgs,-1);
        long s= 0;
        for (int i=0;i<n;i++){
            s +=nums[i];
            if (i<k*2){
                continue;
            }
            avgs[i-k] = (int)(s/(k*2+1));
            s -=nums[i-k*2];
        }
        return avgs;
    }
    public int[] getAverages(int[] nums, int k){
        int n = nums.length;
        int l = 2*k+1;
        int[] res = new int [n];
        for (int i =0;i<n;i++){
            res[i] = -1;
        }
        if (l>n){
            return res;
        }
        long sum  = 0;
        for (int i=0;i<l;i++){
            sum +=nums[i];
        }
        res[k] = (int)(sum/l);
        for (int i=l;i<n;i++){
            sum +=nums[i]-nums[i-l];
            res[i-k] = (int)(sum/l);
        }
        return res;
    }
}
class Solution2841{
    public long maxSum(List<Integer> nums, int m, int k){
        Integer[] a = nums.toArray(new Integer[0]);
        long ans= 0;
        long s =0;
        Map<Integer,Integer> cnt  = new HashMap<>();

        for (int i =0;i<a.length;i++){
            s +=a[i];
            cnt.merge(a[i],1,Integer::sum);
            int left = i-k+1;
            if (left<0){
                continue;
            }
            if (cnt.size()>=m){
                ans = Math.max(ans,s);
            }
            int out = a[left];
            s -=out;
            int c = cnt.get(out);
            if (c>1){
                cnt.put(out,c-1);
            }else {
                cnt.remove(out);
            }

        }
        return ans;
    }
}
class Solution3439 {
    public int maxFreeTime(int eventTime, int k, int[] startTime, int[] endTime) {
        int ans = 0;
        int s = 0;
        for (int i = 0; i <= startTime.length; i++) {
            s += get(i, eventTime, startTime, endTime);
            if (i >= k) {
                ans = Math.max(ans, s); // ✅ 更新最大值
                s -= get(i - k, eventTime, startTime, endTime);
            }
        }
        return ans;
    }

    private int get(int i, int eventTime, int[] startTime, int[] endTime) {
        if (i == 0) {
            return startTime[0]; // 第一个活动前的空闲时间
        }
        int n = startTime.length;
        if (i == n) {
            return eventTime - endTime[n - 1]; // 所有活动结束后的空闲时间
        }
        return startTime[i] - endTime[i - 1]; // 活动之间的空闲时间
    }
}
class Solution1007{
    public int minDominoRotations(int[] tops, int[] bottoms){
        int ans = Math.min(minRot(tops,bottoms,tops[0]),minRot(tops,bottoms,bottoms[0]));
        return ans == Integer.MAX_VALUE?-1:ans;
    }
    private int minRot(int[] tops,int[] bottoms,int target){
        int totop = 0;
        int tobottom = 0;
        for (int i =0;i< tops.length;i++){
            int x = tops[i];
            int y = bottoms[i];
            if (x!=target&&y!=target){
                return Integer.MAX_VALUE;
            }
            if (x!=target){
                totop++;
            }
            else if (y!=target){
                tobottom++;
            }
        }
        return Math.min(tobottom,totop);
    }
}
class Solution438{
    public List<Integer> findAnagramsA(String s, String p){
        List<Integer> ans = new ArrayList<>();
        if (s.length()<p.length()) return ans;
        int[] cnt = new int[26];
        for (char c:p.toCharArray()){
            cnt[c-'a']++;
        }
        int left = 0,right  = 0,required= p.length();
        while (right<s.length()){
            int c = s.charAt(required)-'a';
            if (cnt[c]>0){
                required--;
            }
            cnt[c]--;
            right++;
            if (required==0){
                ans.add(left);
            }
            if (right-left==p.length()){
                int l = s.charAt(left)-'a';
                if (cnt[l]>=0){
                    required++;
                }
                cnt[l]++;
                left++;
            }
        }
        return ans;
    }
    public List<Integer> findAnagrams(String s, String p){
        List<Integer> ans = new ArrayList<>();
        int[] cntP = new int[26];//p种每种字母的出现次数
        int[] cntS = new int[26];//子串每种字母的出现次数
        for (char c:p.toCharArray()){
            cntP[c-'a']++;
        }
        for (int right=0;right<s.length();right++){
            cntS[s.charAt(right)-'a']++;
            int left = right-p.length()+1;
            if (left<0){
                continue;
            }
            if (Arrays.equals(cntP,cntS)){
                ans.add(left);
            }
            cntS[s.charAt(left)-'a']--;
        }
        return ans;
    }
}
class Solution567{
    public boolean checkInclusionA(String s1, String s2){
        int n  = s1.length();
        int m = s2.length();
        if (n>m){
            return false;
        }
        int[] cnt = new int[26];
        for (char c:s1.toCharArray()){
            cnt[c-'a']++;
        }
        int[] cur = new int[26];
        for (int i =0;i<n;i++){
            cur[s2.charAt(i)-'a']++;
        }
        if (check(cnt,cur)){
            return true;
        }
        for (int i =n;i<m;i++){
            cur[s2.charAt(i)-'a']++;
            cur[s2.charAt(i-n)-'a']--;
            if (check(cnt,cur)){
                return true;
            }
        }
        return false;
    }
    private boolean check(int[] cnt1, int[] cnt2){
        for (int i=0;i<26;i++){
            if (cnt1[i]!=cnt2[i]){
                return false;
            }
        }
        return true;
    }
    public boolean checkInclusion(String s1, String s2){
        int n =s1.length(),m= s2.length();
        if (n>m) return false;
        int[] cntS1 = new int[26];
        int[] cntS2 = new int[26];

        for (char c:s1.toCharArray()){
            cntS1[c-'a']++;
        }
        for (int right=0;right<m;right++){
            cntS2[s2.charAt(right)-'a']++;
            int left =right-n+1;
            if (left<0){
                continue;
            }
            if (Arrays.equals(cntS1,cntS2)){
                return true;
            }
            cntS2[s2.charAt(left)-'a']--;
        }
        return false;
    }
}
class Solution1128{
    public int numEquivDominoPairs(int[][] dominoes){
        int ans  =0;
        int[][] cnt = new int[10][10];
        for (int[] d :dominoes){
            int a = Math.min(d[0],d[1]);
            int b = Math.max(d[0],d[1]);
            ans += cnt[a][b]++;
        }
        return ans;
    }
}
class Solution3A{
    public int lengthOfLongestSubstring(String s){
        int n = s.length(),ans = 0;
        Map<Character,Integer> map = new HashMap<>();
        for (int end= 0,start = 0;end<n;end++){
            char alpha = s.charAt(end);
            if (map.containsKey(alpha)){
                start = Math.max(map.get(alpha),start);
            }
            ans = Math.max(ans,end-start+1);
            map.put(s.charAt(end),end+1);
        }
        return ans;
    }
}
class Solution3090{
    public int maximumLengthSubstring(String S){
        char[] s=S.toCharArray();
        int ans = 0;
        int left = 0;
        int[] cnt = new int [26];
        for (int i=0;i<s.length;i++){
            int b = s[i]-'a';
            cnt[b]++;
            while (cnt[b]>2){
                cnt[s[left++]-'a']--;
            }
            ans = Math.max(ans,i-left+1);
        }
        return ans;
    }
}
class Solution1493{
    public int longestSubarray(int[] nums){
        int n  = nums.length;
        int l=-1,zero = -1;
        int ans = 0;
        for (int i =0;i<n;i++){
            if (nums[i]==0){
                l = zero+1;
                zero = i;
            }
            ans = Math.max(ans,i-l);
        }
        return Math.min(ans,n-1);
    }
}

class Solution3536 {
    public int maxProduct(int n) {
        int[] count = new int[10]; // 用于统计每个数字出现的次数
        List<Integer> digits = new ArrayList<>();

        // 提取每一位数字，并统计次数
        while (n > 0) {
            int digit = n % 10;
            digits.add(digit);
            count[digit]++;
            n /= 10;
        }

        int ans = 0;
        for (int i = 0; i < digits.size(); i++) {
            for (int j = 0; j < digits.size(); j++) {
                int a = digits.get(i);
                int b = digits.get(j);

                if (a == b && count[a] < 2) continue; // 相同数字必须出现至少两次
                ans = Math.max(ans, a * b);
            }
        }

        return ans;
    }
}
class Solution790{
    private static final int MOD = 1_000_000_007;
    public int numTilings(int n ){
        if (n==1){
            return 1;
        }
        long [] f = new long[n+1];
        f[0] = f[1] = 1;
        f[2] = 2;
        for (int i =3;i<=n;i++){
            f[i] = (f[i-1]*2+f[i-3])%MOD;
        }
        return (int) f[n];
    }
}
class Solution940{
    public int totalFruit(int[] fruits){
        int ans=  0;
        int left  =0;
        Map<Integer,Integer> cnt = new HashMap<>();
        for (int right = 0;right<fruits.length;right++){
            cnt.merge(fruits[right],1,Integer::sum);
            while (cnt.size()>2){
                int out = fruits[left];
                cnt.merge(out,-1,Integer::sum);
                if (cnt.get(out)==0){
                    cnt.remove(out);
                }
                left++;
            }
            ans = Math.max(ans,right-left+1);
        }
        return ans;
    }
}
class Solution1695{
    public int maximumUniqueSubarrayA(int[] nums){
        int n = nums.length;
        Map<Integer,Integer> map = new HashMap<>();
        int l =0,sum = 0,ans = 0;
        for (int i =0;i<n;i++){
            sum +=nums[i];
            if(map.merge(nums[i],1,Integer::sum)==1){
                ans = Math.max(ans,sum);
            }else {
                while (true){
                    if (map.get(nums[l])==1){
                        sum -=nums[l];
                    }else {
                        sum -=nums[l];
                        map.merge(nums[l++],-1,Integer::sum);
                        break;
                    }
                }
            }
        }
        return ans;
    }
    public int maximumUniqueSubarray(int[] nums){
        int sum = 0;
        int max = 0;
        HashSet<Integer> set = new HashSet<Integer>();
        int left = 0;
        for (int right = 0;right<nums.length;right++){
            while (set.contains(nums[right])){
                sum -=nums[left];
                set.remove(nums[left]);
                left++;
            }
            set.add(nums[right]);
            sum +=nums[right];
            max = Math.max(max,sum);
        }
        return max;
    }
}
class Solution1920{
    public int[] buildArray(int[] nums){
        int[] ans = new int[nums.length];
        for (int i =0;i<nums.length;i++){
            ans[i] = nums[nums[i]];
        }
        return ans;
    }
}
class Solution2958{
    public int maxSubarrayLength(int[] nums, int k){
        int ans = 0, left= 0;
        Map<Integer,Integer> cnt = new HashMap<>();
        for (int right = 0;right<nums.length;right++){
            cnt.merge(nums[right],1,Integer::sum);
            while (cnt.get(nums[right])>k){
                cnt.merge(nums[left++],-1,Integer::sum);
            }
            ans = Math.max(ans,right-left+1);
        }
        return ans;
    }
}
class Solution2024{
    public int maxConsecutiveAnswers(String answerKey, int k){
        char[] s= answerKey.toCharArray();
        int ans = 0;
        int left  =0;
        int[] cnt = new int [2];
        for (int right = 0;right<s.length;right++){
            cnt[s[right] >>1&1]++;
            while (cnt[0]>k&&cnt[1]>k){
                cnt[s[left++]>>1&1]--;
            }
            ans  = Math.max(ans,right-left+1);
        }
        return ans;
    }
}
class Solution1004{
    public int longestOnes(int[] nums, int k){
        int ans=0,cnt0= 0,left=0;
        for (int right = 0;right<nums.length;right++){
            cnt0 +=1-nums[right];
            while (cnt0>k){
                cnt0 -=1-nums[left++];
            }
            ans = Math.max(ans,right-left+1);
        }
        return ans;
    }
}
class Solution1358{
    public int numberOfSubstrings(String S){
        char[] s = S.toCharArray();
        int ans = 0;
        int left = 0;
        int[] cnt = new int[3];
        for (char c:s){
            cnt[c-'a']++;
            while (cnt[0]>0&&cnt[1]>0&&cnt[2]>0){
                cnt[s[left]-'a']--;//左边端口值的出现次数减少
                left++;//收缩窗口
            }
            ans +=left;
        }
        return ans;
    }
}
class Solution2962A{
    public long countSubarrays(int[] nums, int k){
        int mx = 0;
        for (int x:nums){
            mx = Math.max(mx,x);
        }
        long ans  =0;
        int cntMx=  0,left = 0;
        for (int x:nums){
            if (x==mx){
                cntMx++;
            }
            while (cntMx==k){
                if (nums[left]==mx){
                    cntMx--;
                }
                left++;
            }
            ans +=left;
        }
        return ans;
    }
}
class Solution2926B{
    public long countSubarrays(int[] nums, int k){
        int mx = Arrays.stream(nums).max().getAsInt();
        int n = nums.length;
        long ans = 0;
        int cnt =0,left = 0;
        for (int x:nums){
            while (left<n&&cnt<k){
                cnt +=nums[left++]==mx?1:0;
            }
            if (cnt<k){
                break;
            }
            ans +=n-left+1;
            cnt -=x==mx?1:0;
        }
        return ans;
    }
}
class Solution3325{
    int numberOfSubstrings(String S, int k){
        char[] s= S.toCharArray();
        int ans= 0 ;
        int left = 0;
        int [] cnt = new int[26];
        for (char c:s){
            cnt[c-'a']++;
            while (cnt[c-'a']>=k){
                cnt[s[left]-'a']--;
                left++;
            }
            ans +=left;
        }
        return ans;
    }
}
class Solution2799A{
    public int countCompleteSubarrays(int[] nums){
        Set<Integer> set = new HashSet<>();
        for (int num:nums){
            set.add(num);
        }
        int distinct = set.size();
        Map<Integer,Integer> windows = new HashMap<>();
        int result = 0;
        int left = 0;
        for (int right =0;right<nums.length;right++){
            windows.merge(nums[right],1,Integer::sum);
            while (windows.size()==distinct){
                int leftnum = nums[left];
                int newCount = windows.get(leftnum)-1;
                if (newCount==0){
                    windows.remove(leftnum);
                }else {
                    windows.put(leftnum,newCount);
                }
                left++;
            }
            result +=left;
        }
        return result;
    }
}
class Solution2537{
    public long countGood(int[] nums, int k){
        long ans  = 0;
        Map<Integer,Integer> cnt = new HashMap<>();
        int p = 0;
        int left =0;
        for (int x:nums){
            int c = cnt.getOrDefault(x,0);
            p +=c;
            cnt.put(x,c+1);
            while (p>=k){
                x = nums[left];
                c = cnt.get(x);
                p -=c-1;
                cnt.put(x,c-1);
                left++;
            }
            ans +=left;
        }
        return ans;
    }
    public long countGoodA(int[] nums, int k){
        Map<Integer,Integer> cnt = new HashMap<>();
        long ans = 0;
        long p  = 0;
        int left = 0;
        for (int right=  0;right<nums.length;right++){
            int c = cnt.getOrDefault(nums[right],0);
            cnt.put(nums[right],c+1);

            }
return ans;
        }
}
class Solution713{
    public int numSubarrayProductLessThanK(int[] nums, int k){
        if (k<=1){
            return 0;
        }
        int ans= 0;
        int x = 1;
        int left  = 0;
        for (int right = 0;right<nums.length;right++){
            x *=nums[right];
            while (x>=k){
                x /=nums[left++];
            }
            ans +=right-left+1;
        }
        return ans;
    }
}
class Solution3258{
    public int countKConstraintSubstrings(String S, int k){
        char[] s = S.toCharArray();
        int ans=0,left = 0;
        int[] cnt = new int[2];
        for (int right =0;right<s.length;right++){
            cnt[s[right]&1]++;
            while (cnt[0]>k&&cnt[1]>k){
                cnt[s[left]&1]--;
                left++;
            }
            ans +=right-left+1;
        }
        return ans;
    }
}
class Solution2918 {
    public long minSum(int[] nums1, int[] nums2) {
        long sa = 0, za = 0, sb = 0, zb = 0;
        for (int x : nums1) {
            sa += x;
            if (x == 0) za++;
        }
        for (int x : nums2) {
            sb += x;
            if (x == 0) zb++;
        }
        if (za == 0 && zb == 0) return sa == sb ? sa : -1;
        if (za == 0) {
            if (sb + zb > sa) return -1;
            return sa;
        }
        if (zb == 0) {
            if (sa + za > sb) return -1;
            return sb;
        }
        return Math.max(sa + za, sb + zb);
    }
}
class Solution2302A{
    public long countSubarrays(int[] nums, long k){
        long ans=0,sum=0;
        int left = 0;
        for (int right = 0;right<nums.length;right++){
            sum +=nums[right];
            while (sum*(right-left+1)>=k){
                sum -=nums[left++];

            }
            ans +=right-left+1;
        }
        return ans;
    }

}
class Solution2762{
    public long continuousSubarrays(int[] nums){
        long ans = 0;
        TreeMap<Integer,Integer> t = new TreeMap<>();
        int left = 0;
        for (int right = 0;right<nums.length;right++){
            t.merge(nums[right],1,Integer::sum);
            while (t.lastKey()-t.firstKey()>2){
                int out = nums[left];
                int c = t.get(out);
                if (c==1){
                    t.remove(out);
                }else {
                    t.put(out,c-1);
                }
                left++;
            }
            ans +=right-left+1;
        }
        return ans;
    }
}
class SolutionLCO68{
    public int beautifulBouquet(int[] flowers, int cnt){
        long ans= 0;
        Map<Integer,Integer> c = new HashMap<>();
        int left = 0;
        for (int right= 0;right<flowers.length;right++){
            int x = flowers[right];
            c.merge(x,1,Integer::sum);
            while (c.get(x)>cnt){
                c.merge(flowers[left++],-1,Integer::sum);
            }
            ans +=right-left+1;
        }
        return (int) (ans%1_000_000_007);
    }
}
class Solution1550{
    public boolean threeConsecutiveOddsA(int[] arr){
        int len = arr.length,left=0;
        if (len<3) return false;
        for (int right = 0;right>len;right++){
            if (arr[right]%2==0){
                left =right+1;
                continue;
            }
            if (right-left==2) return true;
        }
        return false;
    }
    public boolean threeConsecutiveOdds(int[] arr){
        for (int i=2;i<arr.length;i++){
            if (arr[i - 2] % 2 != 0 && arr[i - 1] % 2 != 0 && arr[i] % 2 != 0){
                return true;
            }
        }
        return false;
    }
}
class Solution930{
    public int numSubarraysWithSumA(int[] nums, int goal){
        int ans1 = 0, left1=0,left2=0,ans2=0;
        int sum1=0,sum2=0;
        for (int right = 0;right<nums.length;right++){
            sum1 +=nums[right];
            while (sum1>=goal&&left1<=right){
                sum1 -=nums[left1++];
            }
            ans1 +=left1;
            sum2 +=nums[right];
            while (sum2>=goal+1&&left2<=right){
                sum2 -=nums[left2++];
            }
            ans2 +=left2;
        }
        return ans1-ans2;
    }
    public int numSubarraysWithSum(int[] nums, int goal){
        return atMost(nums,goal)-atMost(nums,goal+1);
    }
    private int atMost(int[] nums, int goal){
        int ans = 0,left = 0,sum = 0;
        for (int right = 0;right<nums.length;right++){
            sum +=nums[right];
            while (sum>=goal&&left<=right){
                sum -=nums[left++];
            }
            ans +=left;
        }
        return ans;
    }
}
class Solution1248{
    public int numberOfSubarraysA(int[] nums, int k){
        int ans = 0,left  =0;
        int cnt = 0,count = 0;
        for (int right= 0;right<nums.length;right++){
            if (nums[right]%2!=0){
                cnt++;
            }
           if (cnt==k&&left<=right){
                count++;
            }
           if (cnt>k){
               while (nums[left]%2!=0){
                   cnt--;
               }
               left++;
           }
        }
        return count;
    }
    public int numberOfSubarrays(int[] nums, int k) {
        return count(nums,k)-count(nums,k-1);
    }
    private int count(int[] nums,int k ){
        int ans = 0,left  = 0;
        for (int right = 0;right<nums.length;right++){
            if (nums[right] % 2 != 0) k--;
            while (k<0){
                if (nums[left++]%2!=0) k++;
            }
            ans +=right-left+1;
        }
        return ans;
    }
}
class Solution2094{
    public int[] findEvenNumbers(int[] digits){
        int[] cnt = new int[10];
        for (int d:digits){
            cnt[d]++;
        }
        List<Integer> ans = new ArrayList<>();
        for (int i =100;i<1000;i+=2){
            int[] c = new int[10];
            for(int x = i;x>0;x/=10){
                int d = x%10;
                if (++c[d]>cnt[d]){
                    continue;
                }
            }
            ans.add(i);
        }
        return ans.stream().mapToInt(i->i).toArray();
    }
}
class Solution344{
    public void reverseString(char[] s){
        int n = s.length;
        for (int left=0,right=n-1;left<right;left++,right--){
            char tmp = s[left];
            s[left] = s[right];
            s[right] = tmp;
        }
    }
}
class Solution125{
    public boolean isPalindrome(String s){
        int n = s.length();
        int left = 0;
        int right = n-1;
        while (left<right){
            if (!Character.isLetterOrDigit(s.charAt(left))){
                left++;
            }else if (!Character.isLetterOrDigit(s.charAt(right))){
                right--;
            }else if (Character.toLowerCase(s.charAt(left))==Character.toLowerCase(s.charAt(right))){
                left++;
                right--;
            }else {
                return false;
            }
        }
        return true;
    }
}
class Solution1750{
    public int minimumLength(String S){
        char[] s = S.toCharArray();
        int n = s.length;
        int left = 0,right = n-1;
        while (left<right&&s[left]==s[right]){
            char c = s[left];
            while (left<=right&&s[left]==c) left++;
            while (left<=right&&s[right]==c) right--;
        }
        return right-left+1;
    }
}
class Solution3335{
    public int lengthAfterTransformations(String s, int t){
        int mod = (int) 1e9+7;
        long[] book = new long[26];
        long ret = s.length();
        for (char c:s.toCharArray()){
            book[c-'a']++;
        }
        for (int i=0;i<t;i++){
            int ida = 25-(i%26);
            if (book[ida]>0){
                int idb = (ida+1) % 26;
                book[idb] = (book[idb] + book[ida]) % mod;
                ret= (ret+book[ida])%mod;
            }
        }
        return (int) ret;
    }
}
class Solution2105{
    public int minimumRefill(int[] plants, int capacityA, int capacityB){
        int ans = 0;
        int a = capacityA;
        int b = capacityB;
        int i =0,j = plants.length-1;
        while (i<j){
            if (a<plants[i]){
                ans++;
                a = capacityA;
            }
            a -=plants[i++];
            if (b<plants[j]){
                ans++;
                b  = capacityB;
            }
            b -=plants[j--];
        }
        if (i==j&&Math.max(a,b)<plants[i]){
            ans++;
        }
        return ans;
    }
}
class Solution977{
    public int[] sortedSquares(int[] nums){
        int n = nums.length;
        int[] ans = new int[n];
        int i =0,j=n-1;
        for (int p =n-1;p>=0;p--){
            int x = nums[i]* nums[i];
            int y = nums[j]*nums[j];
            if (x>y){
                ans[p] = x;
                i++;
            }else {
                ans[p] = y;
                j--;
            }
        }
        return ans;
    }
}
class Solution658B{
    public List<Integer> findClosestElementsA(int[] arr, int k, int x){
        int n = arr.length;
        List<Integer> list = new ArrayList<>();
        int left = 0,right = n-k;
        while (left<=right){
            int mid = (left+right)>>>1;
            if (mid+k<n&&x-arr[mid]>arr[mid+k]-x){
                left = mid+1;
            }else {
                right = mid-1;
            }
        }
        for (int i =left;i<left+k;i++){
            list.add(arr[i]);
        }
        return list;
    }
    public List<Integer> findClosestElements(int[] arr, int k, int x){
        int n = arr.length;
        List<Integer> list = new ArrayList<>();
        int left = 0,right = n-1;
        int del  = n-k;
        while (del>0){
            if (x-arr[left]>arr[right]-x){
                left++;
            }else {
                right--;
            }
            del--;
        }
        for (int i =left;i<left+k;i++){
            list.add(arr[i]);
        }
        return list;
    }
}
class Solution1471{
    public int[] getStrongest(int[] arr, int k){
        int n  = arr.length;
        Arrays.sort(arr);
       int m = arr[(n-1)/2];
       int left = 0,right = n-1;
       int[] ans = new int[k];
      while (k-->0){
           if (m-arr[left]>arr[right]-m){
               ans[k] = arr[left++];
           }else {
               ans[k] = arr[right--];
           }
       }
       return ans;
    }
}
class Solution167{
    public int[] twoSum(int[] numbers, int target){
        int left = 0,right = numbers.length-1;
        while (true){
            int s= numbers[left]+numbers[right];
            if (s==target){
                return new int[]{left+1,right+1};
            }
            if (s>target){
                right--;
            }else {
                left++;
            }
        }
    }
}
class Solution2824{
    public int countPairs(List<Integer> nums, int target){
        Collections.sort(nums);
        int ans = 0,left = 0,right = nums.size()-1;
        while (left<right){
            if (nums.get(left)+nums.get(right)<target){
                ans +=right-left;
                left++;
            }else {
                right--;
            }
        }
        return ans;
    }
}
class Solutionlcp28{
    public int purchasePlans(int[] nums, int target){
        Arrays.sort(nums);
        int ans = 0,left = 0,right = nums.length-1;
        while (left<right){
            if (nums[left]+nums[right]<target){
                ans +=right-left;
                left++;
            }else {
                right--;
            }
            ans %=1_000_000_007;
        }
        return ans;
    }
}
class Solution2900{
    public List<String> getLongestSubsequence(String[] words, int[] groups){
        List<String> ans = new ArrayList<>();
        int n  = groups.length;
        for (int i =0;i<n;i++){
            if (i==n-1||groups[i]!=groups[i+1]){
                ans.add(words[i]);
            }
        }
        return ans;
    }
}
class Solution1616{
    private boolean isPalindrome(String s, int i, int j){
        while (i<j&&s.charAt(i)==s.charAt(j)){
            ++i;
            --j;
        }
        return i>=j;
    }
    private boolean check(String a, String b){
        int i =0,j=a.length()-1;
        while (i<j&&a.charAt(i)==b.charAt(j)){
            ++i;
            --j;
        }
        return isPalindrome(a,i,j)||isPalindrome(b,i,j);
    }
    public boolean checkPalindromeFormation(String a, String b){
        return check(a,b)||check(b,a);
    }
}
class Solution611{
    public int triangleNumber(int[] nums){
        Arrays.sort(nums);
        int n = nums.length;
        int ans=  0;
        for (int i =0;i<n;i++){
            int a = nums[i];
            if (a==0){
                continue;
            }
            int j = i+1;
            for (int k=i+2;k<n;k++){
                while (nums[k]-nums[j]>=a){
                    j++;
                }
                ans +=k-j;
            }
        }
        return ans;
    }

}
class Solution581{
    public int findUnsortedSubarray(int[] nums){
        int n  = nums.length;
        int max = Integer.MIN_VALUE;
        int min  = Integer.MAX_VALUE;
        int left=-1,right = -1;
        for (int i = 0;i<n;i++){
            if (nums[i]>=max){
                max = nums[i];
            }else {
                right = i;
            }
        }
        for (int i =n-1;i>=0;i--){
            if (nums[i]<=min){
                min = nums[i];
            }else {
                left = i;
            }
        }
        return right ==-1?0:right-left+1;
    }
}
class Solution1574{
    public int findLengthOfShortestSubarray(int[] arr){
        int n  =arr.length,right = n-1;
        //找到第一个下降的位置
        while (right>0&&arr[right-1]<=arr[right]){
            right--;
        }
        if (right==0) return 0;
        int ans = right;
        for (int left = 0;left<n ; ++left){
            if (left>0&&arr[left]<arr[left-1]) break;//保证left是递增的
            while (right<n&&arr[right]<arr[left]){
                right++;//找到right保证arr[left]<=arr[right]
            }
            //删除 arr[left+1..right-1] 的长度为 right - left - 1
            ans = Math.min(ans,right-left-1);
        }
        return ans;
    }
}
class Solution1793{
    public int maximumScore(int[] nums, int k){
        int n  = nums.length;
        int ans = nums[k],minH = nums[k];
        int i=k,j=k;
        for (int t = 0;t<n-1;t++){
            if (j==n-1||i>0&&nums[i-1]>nums[j+1]){
                minH = Math.min(minH,nums[--i]);
            }else {
                minH = Math.min(minH,nums[++j]);
            }
            ans = Math.max(ans,minH*(j-i+1));
        }
        return ans;
    }
}
class Solution27A{
    public int removeElement(int[] nums, int val){
        int stackSize = 0;
        for (int x:nums){
            if (x!=val){
                nums[stackSize++] =x;
            }
        }
        return stackSize;
    }
}
class Solution75A{
    public void sortColors(int[] nums){
        int p0 = 0;
        int p1 = 0;
        for (int i=0;i<nums.length;i++){
            int x = nums[i];
            nums[i]  =2;
            if (x<=1){
                nums[p1++] = 1;
            }
            if (x==0){
                nums[p0++] = 0;
            }
        }
    }
}
class Solution26A{
    public int removeDuplicates(int[] nums){
        int k = 1;
        for (int i =1;i<nums.length;i++){
            if (nums[i]!=nums[i-1]){
                nums[k++] = nums[i];
            }
        }
        return k;
    }
}
class Solution80A{
    public int removeDuplicates(int[] nums){
        int stackSize = 2;
        for (int i =2;i<nums.length;i++){
            if (nums[i]!=nums[stackSize-2]){
                nums[stackSize++] = nums[i];
            }
        }
        return Math.min(stackSize,nums.length);
    }
}
class Solution283{
    public void moveZeroes(int[] nums){
        int stackSize = 0;
        for (int x:nums){
            if (x!=0){
                nums[stackSize++] = x;
            }
        }
        Arrays.fill(nums,stackSize,nums.length,0);
    }
}
class Solution905{
    public int[] sortArrayByParity(int[] nums){
        int i =0,j=nums.length-1;
        while (i<j){
            if (nums[i]%2==0){
                i++;
            }
            else if (nums[j]%2==1){
                j--;
            }else {
                int tmp = nums[i];
                nums[i] = nums[j];
                nums[j] = tmp;
                i++;
                j--;
            }
        }
        return nums;
    }
}
class Solution922{
    public int[] sortArrayByParityII(int[] nums){
        int i=0;int j =1;
        while (i<nums.length){
            if (nums[i]%2==0){
                i +=2;
            }else if (nums[j]%2==1){
                j +=2;
            }
            else {
                int tmp = nums[i];
                nums[i] = nums[j];
                nums[j] = tmp;
                i +=2;
                j+=2;
            }
        }
        return nums;
    }
}
class Solution1089{
    public void duplicateZeros(int[] arr){
        int n  = arr.length;
        int countZeros = 0;
        for (int i =0;i<n;i++){
            if (arr[i] ==0){
                countZeros++;
            }
        }
        int i = n-1;
        int j = n+countZeros-1;
        while (i>=0){
            if (arr[i]==0){
                if (j<n) arr[j] = 0;
                j--;
            }
            if (j<n){
                arr[j] = arr[i];
            }
            i--;
            j--;
        }
    }
}
class Solution442{
    public List<Integer> findDuplicates(int[] nums){
        List<Integer> duplicates = new ArrayList<Integer>();
        int n = nums.length;
        for (int i=0;i<n;i++){
            int num  = nums[i];
            int index = Math.abs(num)-1;
            if (nums[index]>0){
                nums[index] -=nums[index];
            }else {
                duplicates.add(index+1);
            }
        }
        return duplicates;
    }
}
class Solution448{
    public List<Integer> findDisappearedNumbers(int[] nums){
        List<Integer> res = new ArrayList<Integer>();
        for (int i =0;i< nums.length;++i){
            int index= Math.abs(nums[i])-1;
            if (nums[index]>0){
                nums[index] *=-1;
            }
        }
        for (int i=0;i<nums.length;++i){
            if (nums[i]>0){
                res.add(i+1);
            }
        }
        return res;
    }
}
class  Solution2109{
    public String addSpaces(String s, int[] spaces){
        StringBuilder ans = new StringBuilder(s.length()+spaces.length);
        int j= 0;
        for (int i = 0;i<s.length();i++){
            if (j<spaces.length&&spaces[j]==i){
                ans.append(' ');
                j++;
            }
            ans.append(s.charAt(i));
        }
        return ans.toString();
    }
}
class Solution2540{
    public int getCommon(int[] nums1, int[] nums2){
        int i =0,j=0;
        while (i<nums1.length&&j<nums2.length){
            int a = nums1[i],b = nums2[j];
            if (a==b) return a;
            if (a<b) i++;
            else j++;
        }
        return -1;
    }
}
class Solution88{
    public void merge(int[] nums1, int m, int[] nums2, int n){
        int p1 = m-1;
        int p2 = n-1;
        int p = m+n-1;
        while (p2>=0){
            if (p1>=0&&nums1[p1]>nums2[p2]){
                nums1[p--] = nums1[p1--];
            }
            else {
                nums1[p--] = nums2[p2--];
            }
        }
    }
}
class SolutionLCP88{
    public int breakfastNumber(int[] staple, int[] drinks, int x){
        Arrays.sort(staple);
        Arrays.sort(drinks);
        int mod = 1_000_000_007;
        int res = 0;
        int j  = drinks.length-1;

        for (int i=0;i<staple.length;i++){
            if (staple[i]>x) break;
            while (j>=0&&staple[i]+drinks[j]>x){
                j--;
            }
            res = (res+(j+1))%mod;
        }
        return res;
    }
}
class Solution1855 {
    public int maxDistance(int[] nums1, int[] nums2) {
        int p1 = 0;
        int p2 = 0;
        int res = 0;
        while (p1 < nums1.length && p2 <nums2.length){
            if(nums1[p1] > nums2[p2]){  //无效
                if(p1 == p2){
                    p1++;
                    p2++;
                }else p1++;
            }else {     //有效
                res =Math.max(res,p2-p1);
                p2++;
            }
        }
        return res;
    }
}
class Solution925{
    public boolean isLongPressedName(String name, String typed){
        int i=0,j=0;
        while (j<typed.length()){
            if (i<name.length()&&typed.charAt(j) ==name.charAt(i)){
                i++;
                j++;
            }else if (j>0&&typed.charAt(j)==typed.charAt(j-1)){
                j++;
            }else {
                return false;
            }
        }
        return i ==name.length();
    }
}
class Solution2337{
    public boolean canChange(String start, String target){
        if (!start.replace("_","").equals(target.replace("_",""))){
            return false;
        }
        for (int i =0,j=0;i<start.length();i++){
            if (start.charAt(i)=='_'){
                continue;
            }
            while (target.charAt(j)=='_'){
                j++;
            }
            if (i!=j&&(start.charAt(i)=='L')==(i<j)){
                return false;
            }
            j++;
        }
        return true;
    }
}


