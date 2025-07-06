

import javax.swing.*;
import java.util.*;


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
class Solution3AA{
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
class Solution392{
    public boolean isSubsequence(String s, String t){
        if (s.isEmpty()) return true;
        int i = 0;
        for (char c:t.toCharArray()){
            if (s.charAt(i)==c&&++i==s.length()){
                return true;
            }
        }
        return false;
    }

}
class Solution524{
    public boolean isSubsequence(String t, String s){
        int indext = 0,indexs=0;
        while (indext<t.length()&&indexs<s.length()){
            if (t.charAt(indext)==s.charAt(indexs)){
                indext++;
            }
            indexs++;
        }
        return indext ==t.length();
    }
    public String findLongestWord(String s, List<String> d){
        String result = "";
        for (String t:d){
            if (isSubsequence(t,s)){
                if (result.length()<t.length()||(result.length()==t.length()&&result.compareTo(t)>0)) {
                    result = t;
                }
            }
        }
        return result;
    }
}
class Solution2486{
    public int appendCharacters(String s, String t){
        char[]cp = t.toCharArray();
        char[]cs = s.toCharArray();
        int k =0,n=cp.length;
        for (char c:cs){
            if (k<n&&cp[k]==c) k++;
        }
        return n-k;
    }
}
class Solution2825{
    public boolean canMakeSubsequence(String str1, String str2){
        char[] s1  = str1.toCharArray();
        char[] s2 = str2.toCharArray();
        int i=0,j=0;
        while (i<s1.length&&j<s2.length){
            char a  =s1[i];
            char b = s2[j];
            if (a==b||(a-'a'+1)%26==(b-'a')){
                j++;
            }
            i++;
        }
        return j==s2.length;
    }
}
class Solution1023{
    private boolean check(String s,String t){
        int m = s.length(),n = t.length();
        int i  =0,j=0;
        for (;j<n;++i,++j){
            while (i<m&&s.charAt(i)!=t.charAt(j)&&Character.isLowerCase(s.charAt(i))){
                ++i;
            }
            if (i==m||s.charAt(i)!=t.charAt(j)){
                return false;
            }
        }
        while (i<m&&Character.isLowerCase(s.charAt(i))){
            ++i;
        }
        return i==m;
    }
    public List<Boolean> camelMatch(String[] queries, String pattern){
        List<Boolean> ans  = new ArrayList<>();
        for (String q:queries){
            ans.add(check(q,pattern));
        }
        return ans;
    }
}
class Solution522{
    private boolean isSubseq(String s, String t){
        int i =0;
        for (char c:t.toCharArray()){
            if (s.charAt(i)==c&&++i==s.length()){
                return true;
            }
        }
        return false;
    }
    public int findLUSlength(String[] strs){
        Arrays.sort(strs,(a,b)->b.length()-a.length());
        for (int i =0;i<strs.length;i++){
            boolean isSub = false;
            for (int j  = 0;j<strs.length;j++){
                if (i==j) continue;
                if (isSubseq(strs[i],strs[j])){
                    isSub  = true;
                    break;
                }
            }
            if (!isSub) return strs[i].length();
        }
        return -1;
    }
}
class Solution2942{
    public List<Integer> findWordsContaining(String[] words, char x){
        List<Integer> ans = new ArrayList<>();
        for(int i =0;i< words.length;i++){
            if (words[i].indexOf(x)>=0){
                ans.add(i);
            }
        }
        return ans;
    }
}
class Solution2367{
    public int arithmeticTripletsA(int[] nums, int diff){
        int ans = 0;
        HashSet set = new HashSet<Integer>();
        for (int x:nums) set.add(x);
        for (int x:nums){
            if (set.contains(x-diff)&&set.contains(x+diff))
                ++ans;
        }
        return ans;
    }
    public int arithmeticTriplets(int[] nums, int diff){
        int ans = 0,i=0,j=1;
        for (int x:nums){
            while (nums[j]+diff<x){
                ++j;
            }
            if (nums[j]+diff>x){
                continue;
            }
            while (nums[i]+diff*2<x){
                ++i;
            }
            if (nums[i]+diff*2==x){
                ++ans;
            }

        }
        return ans;
    }
}
class Solution2536{
    public long countFairPairs(int[] nums, int lower, int upper){
        Arrays.sort(nums);
        long ans  = 0;
        int l  = nums.length;
        int r  = nums.length;
        for (int j =0;j<nums.length;j++){
            while (r>0&&nums[r-1]>upper-nums[j]){
                r--;
            }
            while (l>0&&nums[l-1]>=lower-nums[j]){
                l--;
            }
            ans +=Math.min(r,j)-Math.min(l,j);
        }
        return ans;
    }
}
class Solution1446{
    public int maxPower(String s){
        int n  = s.length(),ans = 1;
        for (int i=0;i<n;){
            int j  = i;
            while (s.charAt(i)==s.charAt(j)&&j<n) j++;
            ans = Math.max(ans,j-i);
            i = j;
        }
        return ans;
    }
}
class Solution1869{
    public boolean checkZeroOnes(String s){
        int n  = s.length();
        int max1=  0,max0=0;
        for (int i =0;i<n;){
            int j =i;
            while (j<n&&s.charAt(j)==s.charAt(i)) j++;
            int len=  j-i;

            if (s.charAt(i)=='1'){
                max1 = Math.max(max1,len);
            }else {
                max0=Math.max(max0,len);
            }
            i=j;
        }
        return max1>max0;
    }
}
class Solution2331{
    public int longestPalindrome(String[] words){
        int ans  = 0;
        int n = words.length;
        Map<String,Integer> map = new HashMap<>();
        for (int i =0;i<n;i++){
            StringBuilder sb =  new StringBuilder(words[i]);
            String s = sb.reverse().toString();
            if (map.getOrDefault(s,0)>0){
                ans +=4;
                map.put(s,map.get(s)-1);
            }else {
                map.put(words[i],map.getOrDefault(words[i],0)+1);
            }
        }
        for (String key:map.keySet()){
            if (map.get(key)>0&&key.charAt(0)==key.charAt(1)){
                ans +=2;
                break;
            }
        }
        return ans;
    }
}
class Solution2414{
    public int longestContinuousSubstring(String S){
        char[] s = S.toCharArray();
        int ans = 1;
        int cnt = 1;
        for (int i=1;i<s.length;i++){
            if (s[i-1]+1==s[i]){
                ans = Math.max(ans,++cnt);
            }else {
                cnt = 1;
            }
        }
        return ans;
    }
}
class Solution1957{
    public String makeFancyString(String s){
        StringBuilder res = new StringBuilder();
        for (char ch:s.toCharArray()){
            int n =res.length();
            if (n>=2&&ch==res.charAt(n-1)&&ch==res.charAt(n-2)){
                continue;
            }
            res.append(ch);
        }
        return res.toString();
    }
}
class Solution674{
    public int findLengthOfLCIS(int[] nums){
        int l =0;
        int r  = 0;
        int len = 0;
        while (r<nums.length){
            if (r==l||nums[r-1]<nums[r]){
                len = Math.max(len,r-l+1);
                r++;
            }else {
                l = r;
            }
        }
        return len;
    }
}
class Solution978{
    public int maxTurbulenceSize(int[] arr){
        int ans = 1, left = 0;
        for (int right = 1;right<arr.length;right++){
            int c = Integer.compare(arr[right-1],arr[right]);
            if (right==arr.length-1||c*Integer.compare(arr[right],arr[right+1])!=-1){
                if (c!=0){
                    ans = Math.max(ans,right-left+1);
                }
                left = right;
            }

        }
        return ans;
    }
}
class Solution2110{
    public long getDescentPeriods(int[] prices){
        long ans = 0;
        for (int right = 0,left = 0;right<prices.length;right++){
            while (left!=right&&prices[right]-prices[right-1]!=-1){
                left  =right;
            }
            ans +=right-left+1L;
        }
        return ans;
    }
}
class Solution2765{
    public int alternatingSubarray(int[] nums){
        int ans = -1,i=0,n = nums.length;
        while (i<n-1){
            if (nums[i+1]-nums[i]!=1){
                i++;
                continue;
            }
            int i0 = i;
            i +=2;
            while (i<n&&nums[i]==nums[i-2]){
                i++;
            }
            ans = Math.max(ans,i-i0);
            i--;
        }
        return ans;
    }
}
class Solution70A{
    public int climbStairs(int n ){
        int[] memo  = new int[n+1];
        return dfs(n,memo);
    }
    private int dfs(int i,int[] memo){
        if  (i<=1){
            return 1;
        }
        if  (memo[i]!=0){
            return memo[i];
        }
        return memo[i] = dfs(i-1,memo)+dfs(i-2,memo);
    }

}
class Solution746{
    public int minCostClimbingStairs(int[] cost){
        int n = cost.length;
        int [] memo = new int [n+1];
        Arrays.fill(memo,-1);
        return dfs(n,memo,cost);
    }
    private int dfs(int i ,int[] memo,int[] cost){
        if (i<=1){
            return 0;
        }
        if (memo[i]!=-1){
            return memo[i];
        }
        int res1 = dfs(i-1,memo,cost)+cost[i-1];
        int res2 = dfs(i-2,memo,cost)+cost[i-2];
        return memo[i] = Math.min(res2,res1);
    }
}
class Solution377{
    public int combinationSum4(int[] nums, int target){
        int[] memo = new int[target+1];
        Arrays.fill(memo,-1);
        return dfs(target,nums,memo);
    }
    private int dfs(int i,int[] nums,int[] memo){
        if (i==0) return 1;
        if (memo[i]!=-1){
            return memo[i];
        }
        int res = 0;
        for (int x:nums){
            if (x<=i){
                res +=dfs(i-x,nums,memo);
            }
        }
        return memo[i]  = res;
    }
}
class Solution2466{
    public int countGoodStrings(int low, int high, int zero, int one){
        int ans = 0;
        final int MOD = 1_000_000_007;
        int [] f = new int[high+1];
        f[0]= 1;
        for (int i=1;i<=high;i++){
            if (i>=zero) f[i] = f[i-zero];
            if (i>=one) f[i] = (f[i]+f[i-one])%MOD;
            if (i>=low) ans = (ans+f[i])%MOD;
        }
        return ans;
    }
}
class Solution3AAA{
    public int lengthOfLongestSubstring(String s){
        Map<Character,Integer> cnt = new HashMap<>();
        int left = -1,res = 0,n = s.length();
        for (int right = 0;right<n;right++){
            if (cnt.containsKey(s.charAt(right))){
                left = Math.max(left,cnt.get(s.charAt(right)));
            }
            cnt.put(s.charAt(right),right);
            res = Math.max(res,right-left);
        }
        return res;
    }
}
class LRUCacheA extends LinkedHashMap<Integer,Integer>{
    private int capacity;
    public LRUCacheA(int capacity) {
        super(capacity,0.75F,true);
        this.capacity = capacity;
    }

    public int get(int key) {
        return super.getOrDefault(key,-1);
    }

    public void put(int key, int value) {
        super.put(key,value);
    }
    @Override
    protected boolean removeEldestEntry(Map.Entry<Integer,Integer> eldest){
        return size()>capacity;
    }
}
class Solution206A {
    public ListNode reverseList(ListNode head) {
        ListNode pre = null;
        ListNode cur = head;
        while (cur != null) {
            ListNode nxt = cur.next;
            cur.next = pre;
            pre = cur;
            cur = nxt;
        }
        return pre;
    }
}
class Solution25A{
    public ListNode reverseKGroup(ListNode head, int k){
        int n =0;
        for (ListNode cur = head;cur!=null;cur =cur.next){
            n++;//计数器
        }
        ListNode dummy = new ListNode(0,head),preHead = dummy;
        ListNode pre = null,cur = head;
        for (;n>=k;n-=k){//重复n-k次
            for (int i = 0;i<k;++i){
                ListNode  next = cur.next;
                cur.next = pre;
                pre =cur;
                cur  = next;
            }
            ListNode tail = preHead.next;//反转区间的结尾
            tail.next = cur;//下个区间
            preHead.next = pre;//区间的新head
            preHead = tail;//到达结尾进行下一个
        }
        return dummy.next;
    }
}
class Solution15A{
    public List<List<Integer>> threeSum(int[] nums){
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        int n  = nums.length;
        for (int i =0;i<n-2;i++){
            int x = nums[i];
            if (i>0&&x==nums[i-1]) continue;//跳过重复数组
            if (x+nums[i+1]+nums[i+2]>0) continue;//没负数就别看了
            if (x+nums[n-2]+nums[n-1]<0) continue;//每个正数也不看了
            int j = i+1;
            int k  = n-1;
            while (j<k){
                int s= x+nums[j]+nums[k];
                if (s>0){
                    k--;
                }else if (s<0){
                    j++;
                }else {
                    //ans.add(List.of(x, nums[j], nums[k]));
                    for (j++;j<k&&nums[j]==nums[j-1];j++);//去重
                    for (k--;k>j&&nums[k]==nums[k+1];k--);//去重


                }
            }
        }
        return ans;
    }
}
class Solution53A{
    public int maxSubArray(int[] nums){
        int ans = nums[0];
        int sum = 0;
        for (int x:nums){
            if (sum>0){
                sum +=x;
            }else {
                sum = x;
            }
            ans = Math.max(sum,ans);
        }
        return ans;
    }
}
class Solution912{
    private static final int INSERTION_SORT_THRESHOLD = 7;
    private static final Random RANDOM  = new Random();

    public int[] sortArray(int[] nums){
        int len  = nums.length;
        quickSort(nums,0,len-1);
        return nums;
    }
    private void quickSort(int[] nums,int left,int right){
        if (right-left<=INSERTION_SORT_THRESHOLD){
            insertSort(nums,left,right);
            return;
        }
        int pIndex = partition(nums,left,right);
        quickSort(nums,left,pIndex-1);
        quickSort(nums,pIndex+1,right);
    }
    private void insertSort(int[] nums,int left,int right){
        for (int i = left+1;i<=right;i++){
            int tmp = nums[i];
            int j = i;
            while (j>left&&nums[j-1]>tmp){
                nums[j] = nums[j-1];
                j--;
            }
            nums[j] = tmp;

        }
    }
    private int partition(int[] nums, int left, int right){
        int randomIndex = left+RANDOM.nextInt(right-left+1);
        swap(nums,randomIndex,left);

        int pivot = nums[left];
        int lt = left+1;
        int gt = right;
        while (true) {
            while (lt <= right && nums[lt] < pivot) {
                lt++;
            }

            while (gt > left && nums[gt] > pivot) {
                gt--;
            }

            if (lt >= gt) {
                break;
            }

            // 细节：相等的元素通过交换，等概率分到数组的两边
            swap(nums, lt, gt);
            lt++;
            gt--;
        }
        swap(nums, left, gt);
        return gt;

    }
    private void swap(int[] nums, int index1, int index2) {
        int temp = nums[index1];
        nums[index1] = nums[index2];
        nums[index2] = temp;
    }

}
class Solution21A{
    public ListNode mergeTowLists(ListNode l1, ListNode l2){
        if (l1==null){
            return l2;
        }
        else if (l2==null){
            return l1;
        }
        else if (l1.val<l2.val){
            l1.next  = mergeTowLists(l1.next,l2);
            return l1;
        }else {
            l2.next = mergeTowLists(l1,l2.next);
            return l2;
        }
    }
}
class Solution5A{
    public String longestPalindrome(String s){
        String res = "";
        for (int i =0;i<s.length();i++){
            String s1 = expend(s,i,i+1);
            String s2 = expend(s,i,i);
            res = res.length()>s1.length()?res:s1;
            res = res.length()>s2.length()?res:s2;
        }
        return res;
    }
    private String expend(String s,int l,int r){
        while (l>=0&&r<s.length()&&s.charAt(l)==s.charAt(r)){
            l--;
            r++;
        }
        return s.substring(l+1,r);
    }
}
class Solution102A{
    public List<List<Integer>> levelOrder(TreeNode root){
        List<List<Integer>> res = new ArrayList<>();
        Queue<TreeNode> queue   = new ArrayDeque<>();
        if (root!=null){
            queue.add(root);
        }
        while (!queue.isEmpty()){
            int n = queue.size();
            List<Integer> level = new ArrayList<>();
            for (int i=0;i<n;i++){
                TreeNode node = queue.poll();
                level.add(node.val);
                if (node.left!=null){
                    queue.add(node.left);
                }
                if (node.right!=null){
                    queue.add(node.right);
                }
            }
            res.add(level);
        }
        return res;
    }
}
class Solution1A{
    public int[] twoSum(int[] nums,int target){
        Map<Integer,Integer> map  = new HashMap<>();
        for (int i=0;i<nums.length;i++){
            if (map.containsKey(target-nums[i])){
                return new int[] {map.get(target-nums[i]),i};
            }
            map.put(nums[i],i);
        }
        throw new IllegalArgumentException("No two sum solution");
    }
}
class Solution33A{
    private boolean check(int[] nums, int target, int i){
        int x = nums[i];
        int end = nums[nums.length-1];
        if (x>end){
            return target>end&&x>=target;
        }
        return target>end||x>=target;
    }
    public int search(int[] nums, int target){
        int left = -1;
        int right = nums.length-1;
        while (left+1<right){
            int mid = (left+right)>>>1;
            if (check(nums,target,mid)){
                right = mid;
            }else{
                left = mid;
            }
        }
        return nums[right]==target?right:-1;
    }
}
class Solution200{
    public int numIslands(char[][] grid){
        int ans = 0;
        for (int i =0;i<grid.length;i++){
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j]=='1'){
                    dfs(grid,i,j);
                    ans++;
                }
            }
        }
        return ans;
    }
    private void dfs(char[][] grid,int i,int j){
        if (i<0||i>=grid.length||j<0||j>=grid[0].length||grid[i][j]!='1'){
            return;
        }
        grid[i][j]='2';
        dfs(grid,i,j-1);
        dfs(grid,i,j+1);
        dfs(grid,i-1,j);
        dfs(grid,i+1,j);
    }
}
class Solution2929{
    public  long distributeCandies(int n, int limit){
        if (n>3*limit) return 0;
        long res = 0;
        for (int i=Math.max(0,n-2*limit);i<=Math.min(n,limit);++i){
            res  +=(Math.min(limit,n-i)-Math.max(0,n-i-limit)+1);
        }
        return res;
    }
}
class Solutoion46{
    public List<List<Integer>> permute(int[] nums){
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> path = Arrays.asList(new Integer[nums.length]);
        boolean[] onPath = new boolean[nums.length];
        dfs(0,nums,ans,path,onPath);
        return ans;
     }
     private void dfs(int i,int[] nums,List<List<Integer>> ans,List<Integer> path,boolean[] onPath){
        if (i==nums.length){
            ans.add(new ArrayList<>(path));
            return;
        }
        for (int j=0;j<nums.length;j++){
            if (!onPath[j]){
                path.set(i,nums[j]);
                onPath[j] = true;
                dfs(i+1,nums,ans,path,onPath);
                onPath[j] = false;
            }
        }
     }
}
class Solution88A{
    public void merge(int[] nums1, int m, int[] nums2, int n){
        int p1 = m-1;
        int p2 = n-1;
        int p =m+n-1;
        while (p2>=0){
            if (p1>=0&&nums1[p1]>=nums2[p2]){
                nums1[p--] = nums1[p1--];
            }
            else {
                nums1[p--] = nums2[p2--];
            }
        }
    }
}
class Solution20A{
    public boolean isValid(String s){
        if (s.isEmpty()){
            return true;
        }
        Stack<Character> stack = new Stack<Character>();
        for (char c :s.toCharArray()){
            if (c=='(') stack.push(')');
            else if (c=='{') stack.push('}');
            else if (c=='[') stack.push(']');
            else if (stack.isEmpty()||c!=stack.pop()){
                return false;
            }

        }
        return stack.isEmpty();
    }

}
class Solution135{
    public int candy(int[] ratings){
        int[] left = new int[ratings.length];
        int[] right = new int[ratings.length];
        Arrays.fill(left,1);
        Arrays.fill(right,1);
        for (int i =1;i<ratings.length;i++){
            if (ratings[i]>ratings[i-1]) left[i] = left[i-1]+1;
        }
        int count = left[ratings.length-1];
        for (int i=ratings.length-2;i>=0;i--){
            if (ratings[i]>ratings[i+1]) right[i] = right[i+1]+1;
            count +=Math.max(left[i],right[i]);
        }
        return count;
    }
}
class Solution121{
    public int maxProfit(int[] prices){
        int cost = Integer.MAX_VALUE,profit = 0;
        for (int p:prices){
            cost = Math.min(cost,p);
            profit = Math.max(profit,p-cost);
        }
        return profit;
    }
}
class Solution103A{
    public List<List<Integer>> zigzagLevelOrder(TreeNode root){

        List<List<Integer>> ans = new ArrayList<>();
        Queue<TreeNode> q = new ArrayDeque<>();

        if(root!=null) q.add(root);
        while (!q.isEmpty()){
            int n = q.size();
            List<Integer> vals = new ArrayList<>(n);
            while (n-->0){
                TreeNode node = q.poll();
                vals.add(node.val);
                if (node.left!=null) q.add(node.left);
                if (node.right!=null) q.add(node.right);
            }
            if (ans.size()%2>0) Collections.reverse(vals);
            ans.add(vals);
        }
        return ans;

    }
}
class Solution236{
    public TreeNode lowestCommonAncestor(TreeNode root,TreeNode p,TreeNode q){
        if (root==null||root==q||root==p) return root;
        TreeNode left = lowestCommonAncestor(root.left,p,q);
        TreeNode right = lowestCommonAncestor(root.right,p,q);
        if (left==null) return right;
        if (right==null) return left;
        return root;
    }
}
class Solution141{
    public boolean hasCycle(ListNode head) {
        ListNode fast = head, slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) {
                return true;
            }
        }
        return false;
    }
}
class Solution92{
    public ListNode reverseBetween(ListNode head, int left, int right){
        ListNode dummy = new ListNode(0,head);
        ListNode p0 = dummy;
        for (int i=0;i<left-1;i++){
            p0= p0.next;
        }
        ListNode pre = null;
        ListNode cur = p0.next;
        for (int i = 0;i<right-left+1;i++){
            ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur= next;
        }
        p0.next.next = cur;
        p0.next = pre;
        return dummy.next;
    }
}
class Solution1298{
    private int ans = 0;
    public int maxCandies(int[] status, int[] candies, int[][] keys, int[][] containedBoxes, int[] initialBoxes){
        int[] hasKey = status;
        boolean[] hasBox = new boolean[status.length];
        for (int x:initialBoxes){
            hasBox[x] = true;
        }
        for (int x:initialBoxes){
            if (hasBox[x]&&hasKey[x]==1){
                dfs(x,candies,keys,containedBoxes,hasKey,hasBox);
            }
        }
        return ans;
    }
    private void dfs(int x, int[] candies, int[][] keys, int[][] containedBoxes, int[] hasKey, boolean[] hasBox){
        ans +=candies[x];
        hasBox[x]= false;
        for (int y:keys[x]){
            hasKey[y] = 1;
            if (hasBox[y]){
                dfs(y,candies,keys,containedBoxes,hasKey,hasBox);
            }
        }
        for (int y:containedBoxes[x]){
            hasBox[y] = true;
            if (hasKey[y]==1){
                dfs(y, candies, keys, containedBoxes, hasKey, hasBox);
            }
        }
    }
}
class Solution54A{
    public List<Integer> spiralOrder(int[][] matrix){
        List<Integer> res = new ArrayList<>();
        if (matrix.length==0) return res;
        int l =0,r = matrix[0].length-1;
        int t = 0,b = matrix.length-1;

        while (l<=r&&t<=b){
            for (int i = l;i<=r;i++) res.add(matrix[t][i]);
            t++;
            if (t>b) break;

            for (int i =t;i<=b;i++) res.add(matrix[i][r]);
            r--;
            if (l>r) break;

            for (int i=r;i>=l;i--) res.add(matrix[b][i]);
            b--;
            if (t>b) break;

            for (int i =b;i>=t;i--) res.add(matrix[i][l]);
            l++;
            if (l>r) break;
        }
        return res;
    }
}
class Solution300{
    public int lengthOfLIS(int[] nums){
        int[] res = new int[nums.length];
        int k = 0;
        res[0] = nums[0];
        for (int i  =1;i<nums.length;i++){
            if (nums[i]>res[k]){
                res[++k] = nums[i];
                continue;
            }
            for (int j = 0;j<=k;j++){
                if (res[j]>=nums[i]){
                    res[j] =nums[i];
                    break;
                }
            }
        }
        return k+1;
    }
}
class Solution3403{

    public String answerString(String word, int numFriends) {
        if (numFriends == 1) {
            return word;
        }
        int n = word.length();
        String ans = "";
        for (int i = 0; i < n; ++i) {
            String t = word.substring(i, Math.min(n, i + n - (numFriends - 1)));
            if (ans.compareTo(t) < 0) {
                ans = t;
            }
        }
        return ans;
    }
}
class Solution23A{
    public ListNode mergeKLists(ListNode[] lists){
        int m =  lists.length;
        if (m==0){
            return null;
        }
        for (int step = 1;step<m;step*=2){
            for (int i =0;i<m-step;i+=step*2){
                lists[i] =  meryTwoLists(lists[i],lists[i+step]);
            }
        }
        return lists[0];
    }
    private ListNode meryTwoLists(ListNode list1,ListNode list2){
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;
        while (list1!=null&&list2!=null){
            if (list1.val<list2.val){
                cur.next = list1;
                list1 = list1.next;
            }else {
                cur.next = list2;
                list2 = list2.next;
            }
            cur = cur.next;
        }
        cur.next = list1!=null?list1:list2;
        return dummy.next;
    }
}
class Solution143{
    public ListNode middleNode(ListNode head){
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next!=null&&fast.next.next!=null){
            slow =slow.next;
            fast = fast.next.next;
        }
        return slow;
    }
    public ListNode reverseList(ListNode head){
        ListNode pre = null;
        ListNode cur = head;
        while (cur!=null){
            ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
    public void mergeList(ListNode l1, ListNode l2){
        ListNode l1_temp;
        ListNode l2_temp;
        while (l1!=null&&l2!=null){
            l1_temp = l1.next;
            l2_temp = l2.next;

            l1.next = l2;
            l1 = l1_temp;

            l2.next = l1;
            l2 = l2_temp;
        }
    }
    public void reorderList(ListNode head){
        if (head==null){
            return;
        }
        ListNode mid = middleNode(head);
        ListNode l1 =head;
        ListNode l2 = mid.next;
        mid.next = null;
        l2 = reverseList(l2);
        mergeList(l1,l2);
    }
}
class Solution415{
    public String addStrings(String num1, String num2){
        StringBuilder res = new StringBuilder("");
        int i =  num1.length()-1, j = num2.length()-1,carry = 0;
        while (i>=0||j>=0){
            int n1 = i>=0?num1.charAt(i)-'0':0;
            int n2 = j>=0?num2.charAt(j)-'0':0;
            int tmp = n1+n2+carry;
            carry = tmp/10;
            res.append(tmp%10);
            i--;j--;
        }
        if (carry==1) res.append(1);
        return res.reverse().toString();
    }
}
class Solution56A{
    public int[][] merge(int[][] intervals){
        Arrays.sort(intervals,(p,q)->p[0]-q[0]);
        List<int[]> ans = new ArrayList<>();
        for(int[] p:intervals){
            int m = ans.size();
            if (m>0&&p[0]<=ans.get(m-1)[1]){
                ans.get(m-1)[1] = Math.max(ans.get(m-1)[1],p[1]);
            }else {
                ans.add(p);
            }
        }
        return ans.toArray(new int[ans.size()][]);
    }
}
class Solution160{
    public ListNode getIntersectionNode(ListNode headA, ListNode headB){
        if (headA==null||headB==null) return null;
        ListNode A = headA,B=headB;
        while (A!=B){
            A=(A!=null)?A.next:headB;
            B=(B!=null)?B.next:headA;
        }
        return A;
    }
}
class Solution42A{
    public int trap(int[] height){
        int left = 0,right = height.length-1;
        int preMax= 0,SubMax = 0;
        int ans = 0;
        while (left<right){
            preMax = Math.max(height[left],preMax);
            SubMax = Math.max(height[right],SubMax);
            ans +=preMax<SubMax?preMax-height[left++]:SubMax-height[right--];

        }
        return ans;
    }
    public int trapA(int[] height){
        int index = 0,peak = 0;
        for (int i=0;i<height.length;i++){
            if (height[i]>peak){
                peak = height[i];
                index = i;
            }
        }
        int ans = 0;
        int peakInterL =0;
        for (int i=0;i<index;i++){
            if (height[i]>peakInterL){
                peakInterL = height[i];
            }
            ans +=(peakInterL-height[i]);
        }
        int peakInterR = 0;
        for (int i = height.length-1;i>index;i--){
            if (height[i]>peakInterR){
                peakInterR = height[i];
            }
            ans +=(peakInterR-height[i]);
        }
        return ans;
    }
}
class Solution2334{
    public String robotWithString(String s){
        int n  = s.length();
        char[] sufMin = new char[n+1];
        sufMin[n]  = Character.MAX_VALUE;
        for (int i=n-1;i>=0;i--){
            sufMin[i] = (char) Math.min(sufMin[i+1],s.charAt(i));
        }//
        StringBuilder ans = new StringBuilder();
        Deque<Character> st = new ArrayDeque<>();
        for (int i =0;i<n;i++){
            st.push(s.charAt(i));
            while (!st.isEmpty()&&st.peek()<=sufMin[i+1]){
                ans.append(st.pop());
            }
        }
        return ans.toString();
    }
}
class Solution124{
    private int ans = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        dfs(root);
        return ans;
    }
    private int dfs(TreeNode node){
        if (node==null) return 0;
        int LVal = dfs(node.left);
        int RVal = dfs(node.right);
        ans  = Math.max(ans,LVal+RVal+node.val);
        return Math.max(Math.max(LVal,RVal)+node.val,0);
    }
}
class Solution142{
    public ListNode detectCycle(ListNode head){
        ListNode fast= head,slow  =head;
        while (fast!=null&&fast.next!=null){
            fast = fast.next.next;
            slow  = slow.next;
            if (fast==slow){
                fast = head;
                while (slow!=fast){
                    slow = slow.next;
                    fast = fast.next;
                }
                return fast;
            }
        }
        return null;
    }
}
class Solution93{
    private List<String> ans = new ArrayList<>();
    private List<String> segments = new ArrayList<>();

    public List<String> restoreIpAddresses(String s){
        if (s.length()<4||s.length()>12){
            return ans;
        }
        dfs(s,0,segments);
        return ans;
    }
    private void dfs(String s,int index,List<String> segments){
        if (segments.size()==4){
            if (index==s.length()){
                ans.add(String.join(".",segments));
            }
            return;
        }
        for (int len = 1;len<=3;len++){
            if (index+len>s.length()){
                break;
            }
            String segment = s.substring(index,index+len);
            if (isValid(segment)) {
                segments.add(segment);
                dfs(s, index + len, segments);
                segments.remove(segments.size() - 1);  //恢复现场
            }
        }
    }
    private boolean isValid(String segment){
        if (segment.length() > 1 && segment.startsWith("0")){
            return false;
        }
        int num = Integer.parseInt(segment);
        return num >= 0 && num <= 255;
    }
}
class Solution3170{
    public String clearStars(String S){
        char[] s = S.toCharArray();
        List<Integer>[] stacks = new ArrayList[26];
        Arrays.setAll(stacks,i->new ArrayList<>());
        for (int i=0;i<s.length;i++){
            if(s[i]!='*'){
                stacks[s[i]-'a'].add(i);
                continue;
            }
            for (List<Integer> st:stacks){
                if (!st.isEmpty()){
                    st.remove(st.size()-1);
                    break;
                }
            }
        }
        List<Integer> idx = new ArrayList<>();
        for (List<Integer> st:stacks){
            idx.addAll(st);
        }
        Collections.sort(idx);
        StringBuilder ans = new StringBuilder(idx.size());
        for (int i:idx){
            ans.append(s[i]);
        }
        return ans.toString();
    }
}
class Solution1143{
    private char[] s,t;
    private int[][] memo;
    public int longestCommonSubsequence(String text1, String text2){
        s =text1.toCharArray();
        t = text2.toCharArray();
        int n  =s.length;
        int m = t.length;
        memo = new int[n][m];
        for (int[] row:memo){
            Arrays.fill(row,-1);
        }
        return dfs(n-1,m-1);
    }
    private int dfs(int i ,int j){
        if (i<0||j<0) return 0;
        if (memo[i][j]!=-1) return memo[i][j];
        if (s[i]==t[j]) return memo[i][j] = dfs(i-1,j-1)+1;
        return memo[i][j] = Math.max(dfs(i-1,j),dfs(i,j-1));

    }
    public int longestCommonSubsequenceA(String text1, String text2){
        s =text1.toCharArray();
        t = text2.toCharArray();
        int n  =s.length;
        int m = t.length;
        int[][] f = new int[n+1][m+1];
        for (int i =0;i<n;i++){
            for (int j=0;j<m;j++){
                f[i+1][j+1] = s[i] == t[j]?f[i][j]+1:Math.max(f[i][j+1],f[i+1][j]);
            }
        }
        return f[n][m];
    }
    public int longestCommonSubsequenceB(String text1, String text2){
        char[] t = text2.toCharArray();
        int m = t.length;
        int[] f = new int[m + 1];
        for (char x:text1.toCharArray()){
            int pre=  0;
            for (int j=0;j<m;j++){
                int tmp = f[j+1];
                f[j+1] = x==t[j]?pre+1:Math.max(f[j+1],f[j]);
                pre = tmp;
            }
        }
        return f[m];
    }
}
class  Solution19A{
    public ListNode removeNthFromEnd(ListNode head, int n){
        ListNode dummy = new ListNode(0,head);
        ListNode left = dummy;
        ListNode right = dummy;
        while (n-->0){
            right = right.next;
        }
        while (right.next!=null){
            left = left.next;
            right = right.next;
        }
        left.next = left.next.next;
        return dummy.next;
    }
}
class Solution82A{
    public ListNode deleteDuplicates(ListNode head){
        ListNode dummy = new ListNode(0,head);
        ListNode cur = dummy;
        while (cur.next!=null&&cur.next.next!=null) {
            int val = cur.next.val;
            if (cur.next.next.val==val){
                while (cur.next != null && cur.next.next != null && cur.next.val == val)
                {
                    cur.next= cur.next.next;
                }
            }else {
                cur = cur.next;
            }
        }
        return dummy.next;
    }
}
class Solution386{
    List<Integer> ans  = new ArrayList<>();
    public List<Integer> lexicalOrder(int n){
        for (int i=1;i<=9;i++) dfs(i,n);
        return ans;
    }
    void dfs(int cur,int limit){
        if (cur>limit) return;
        ans.add(cur);
        for (int i=0;i<=9;i++) dfs(cur*10+i,limit);
    }
}
class Solution199{
    public List<Integer> rightSideView(TreeNode root){
        List<Integer> ans = new ArrayList<>();
        dfs(root,0,ans);
        return ans;
    }
    private void dfs(TreeNode root,int depth,List<Integer> ans){
        if (root==null) return ;
        if (depth==ans.size()){
            ans.add(root.val);
        }
        dfs(root.right,depth+1,ans);
        dfs(root.left,depth+1,ans);
    }
}
class Solution94{
    public List<Integer> inorderTraversal(TreeNode root){
        List<Integer> ans = new ArrayList<>();
        dfs(root,ans);
        return ans;
    }
    private void dfs(TreeNode root,List<Integer> ans){
        if (root==null) return;
        dfs(root.left,ans);
        ans.add(root.val);
        dfs(root.right,ans);
    }
}
class Solution704A{
    public int search(int[] nums, int target){
        int left = 0,right = nums.length-1;
        while (left<=right){
            int mid = (left+right)>>>1;
            if (nums[mid]<target){
                left = mid+1;
            }else if (nums[mid]==target){
                return mid;
            }else {
                right = mid-1;
            }
        }
        return -1;
    }
}
class MyQueue {
    private Stack<Integer> A;
    private Stack<Integer> B;

    public MyQueue() {
        A = new Stack<>();
        B = new Stack<>();
    }

    public void push(int x) {
        A.push(x);
    }

    public int pop() {
        int peek =peek();
        B.pop();
        return peek;
    }

    public int peek() {
        if (!B.isEmpty()) return B.peek();
        if (A.isEmpty()) return -1;
        while (!A.isEmpty()){
            B.push(A.pop());
        }
        return B.peek();
    }

    public boolean empty() {
        return A.isEmpty()&&B.isEmpty();
    }
}
class Solution440{
    int ans  = 0;
    int count = 0;
    public int findKthNumber(int n, int k){
        for (int i=1;i<=9;i++) dfs(i,n,k);
        return ans;
    }
    void dfs(int cur,int limit,int k){
        if (cur>limit||ans!=-1) return;
        count++;
        if (count==k){
            ans =cur;
            return;
        }

        for (int i=0;i<=9;i++) dfs(cur*10+i,limit,k);
    }
}
class Solution440A{
    public int findKthNumber(int n, int k){
        int node = 1;
        k--;
        while (k>0){
            int size = countSubtreeSize(n,node);
            if (size<=k){
                node++;
                k -=size;
            }else {
                node *=10;
                k--;
            }
        }
        return node;
    }
    private int countSubtreeSize(int n, int node){
        int size = 0;
        long left = node,right = node+1;
        while (left<=n){
            size +=Math.min(right,n+1)-left;
            left *=10;
            right*=10;
        }
        return size;
    }
}
class Solution165{
    public int compareVersion(String v1, String v2){
        int i=0,j=0;
        int n = v1.length(),m= v2.length();
        while (i<n||j<m){
            int num1 = 0,num2=0;
            while (i<n&&v1.charAt(i)!='.') num1 = num1*10+v1.charAt(i++)-'0';
            while (j<m&&v2.charAt(j)!='.') num2 = num2*10+v2.charAt(j++)-'0';
            if (num1>num2) return 1;
            else if (num1<num2) return -1;
            i++;j++;

        }
        return 0;
    }
}
class Solution148{
    public ListNode sort(ListNode head){
        if (head==null||head.next==null){
            return head;
        }
        PriorityQueue<ListNode> queue = new PriorityQueue<>(Comparator.comparing(n->n.val));
        for (;head!=null;head =head.next){
            queue.offer(head);
        }
        ListNode result =queue.poll(),previous =result;
        for (ListNode node = queue.poll();node!=null;node=queue.poll(),previous=previous.next){
            previous.next = node;
        }
        previous.next = null;
        return result;
    }
}

class Solution148A {
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) return head;

        ListNode s = head, f = head, ps = head;
        while (f != null && f.next != null) {
            f = f.next.next;
            ps = s;
            s = s.next;
        }

        ps.next = null;

        ListNode l = sortList(head);
        ListNode r = sortList(s);

        return merge(l, r);
    }

    ListNode merge(ListNode l, ListNode r) {
        ListNode dummy = new ListNode(0);
        ListNode cur = dummy;

        while (l != null && r != null) {
            if (l.val <= r.val) {
                cur.next = l;
                l = l.next;
            } else {
                cur.next = r;
                r = r.next;
            }
            cur = cur.next;

        }
        if (l == null) {
            cur.next = r;
        } else {
            cur.next = l;
        }
        return dummy.next;
    }
}
class Solution22A{

    private int n;
    private final List<String> ans = new ArrayList<>();
    private char[] path;
    public List<String> generateParenthesis(int n){
        this.n = n;
        path = new char[2*n];
        dfs(0,0);
        return ans;

    }
    private void dfs(int i ,int open){
        if (i==2*n){//填补完成
            ans.add(new String(path));
            return;
        }
        if (open<n){//填补做括号
            path[i] = '(';
            dfs(i+1,open+1);
        }
        if (i-open<open){//填补右括号
            path[i] = ')';
            dfs(i+1,open);
        }
    }
}
class Solution3442{
    public int maxDifference(String s){
        int[] cnt = new int[26];
        for (int c:s.toCharArray()){
            cnt[c-'a']++;
        }
        int max1 = 0;
        int min0 = Integer.MAX_VALUE;
        for (int c:cnt){
            if (c%2>0){
                max1 = Math.max(max1,c);
            } else if (c>0) {
                min0 = Math.min(min0,c);
            }
        }
        return max1-min0;
    }
}
class Solution31A{
    public void nextPermutation(int[] nums){
        int n = nums.length;
        int i = n-2;
        while (i>=0&&nums[i]>=nums[i+1]){
            i--;
        }
        if (i>=0){
            int j = n-1;
            while (nums[j]<=nums[i]){
                j--;
            }
            swap(nums,i,j);
        }
        reverse(nums,i+1,n-1);
    }
    private void swap(int[] nums,int i,int j){
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;

    }
    private void reverse(int[] nums,int left,int right){
        while (left<right){
            swap(nums,left++,right--);
        }
    }
}
class Solution69A{
    public int mySqrt(int x){
        int left = 0, right = Math.min(x, 46340) + 1;
        while (left+1<right){
            int mid = (left+right)>>>1;
            if (mid*mid<=x){
                left = mid;
            }else {
                right = mid;
            }
        }
        return left;
    }
}
class Solution239AA{
    public int[] maxSlidingWindowA(int[] nums, int k ){
        if (nums.length==0||k==0) return new  int[0];
        Deque<Integer> deque = new LinkedList<>();
        int[] res = new int[nums.length-k+1];
        for (int j = 0,i=1-k;j<nums.length;i++,j++){
            if (i>0&&deque.peekFirst()==nums[i-1]){
                deque.removeFirst();
            }
            while (!deque.isEmpty()&&deque.peekLast()<nums[j]){
                deque.removeLast();
                deque.addLast(nums[j]);
            }
            if (i>=0)
                res[i] =deque.peekFirst();
        }
        return res;
    }
    public int[] maxSlidingWindow(int[] nums, int k){
        int n = nums.length;
        int[] ans= new int[n-k+1];
        Deque<Integer> deque= new ArrayDeque<>();
        for (int i =0;i<n;i++){
            while (!deque.isEmpty()&&nums[deque.getLast()]<=nums[i]){
                deque.removeLast();
            }
            deque.addLast(i);

            if (deque.getFirst()<=i-k){
                deque.removeFirst();
            }
            if (i>=k-1){
                ans[i-k+1] = nums[deque.getFirst()];
            }
        }
        return ans;
    }
}
class Solution2E{
    public ListNode addTwoNumbers(ListNode l1, ListNode l2){
        ListNode pre = new ListNode(0);
        ListNode cur  =pre;
        int carry = 0;
        while (l1!=null||l2!=null){
            int x = l1==null?0:l1.val;
            int y = l2==null?0:l2.val;

            int sum  = x+y+carry;


            carry = sum/10;
            sum = sum%10;
            cur.next = new ListNode(sum);

            cur =cur.next;
            if (l1!=null)  l1 =l1.next;
            if (l2!=null)  l2 = l2.next;
        }
        if (carry==1){
            cur.next = new ListNode(carry);
        }
        return pre.next;
    }
}
class Solution8E {
    public int myAtoi(String s) {
        List<Character> ans = new ArrayList<>();
        int sign = 1;
        for (char e:s.toCharArray()){
            if (e==' ') continue;
            if (e=='+') continue;
            if (e=='-') sign = -1;
            if (Character.isDigit(e)){
                ans.add(e);
            }else {
                break;
            }
        }
        if (ans.size()==0) return 0;
        long num = 0;
        for (char c:ans){
            if (sign==1&&num>Integer.MAX_VALUE) return Integer.MAX_VALUE;
            if (sign==-1&&num<Integer.MIN_VALUE) return Integer.MIN_VALUE;
        }
        return (int)(sign*num);
    }
}
class Solution70AA{
    public int climbStairs(int n ){
        int[] memo = new int[n+1];
         return dfs(n,memo);
    }
    private int dfs(int i,int[] memo){
        if (i<=1) return 1;
        if (memo[i]!=0) return memo[i];
        return memo[i] = dfs(i-1,memo)+dfs(i-2,memo);
    }
}
class Solution32A{
    public int longestValidParentheses(String s){
        Deque<Integer> stack = new LinkedList<Integer>();
        int[] dp = new int[s.length()];
        int maxlen = 0;
        for (int i =0;i<s.length();i++){
            if(s.charAt(i)=='('){
                stack.push(i);
            }else if (!stack.isEmpty()&&s.charAt(i)==')'){
                int leftIndex = stack.pop();
                int length = i-leftIndex+1;
                if (leftIndex-1>=0){
                    length+=dp[leftIndex-1];
                }
                dp[i] = length;
                maxlen = Math.max(maxlen,length);
            }
        }
        return maxlen;
    }
}
class Solution322{
    public int coinChange(int[] coins,int amount){
        int[] f = new int[amount+1];
        Arrays.fill(f,Integer.MAX_VALUE/2);
        f[0]= 0;
        for (int x:coins){
            for (int c=x;c<=amount;c++){
                f[c] = Math.min(f[c],f[c-x]+1);
            }
        }
        int ans = f[amount];
        return ans<Integer.MAX_VALUE/2?ans:-1;
    }
}
class Solution43A{
    public String multiply(String num1, String num2){
        if(num1.equals("0")||num2.equals("0")) return "0";
        int[] res = new int[num1.length()+num2.length()];
        for (int i = num1.length()-1;i>=0;i--){
            int n1 = num1.charAt(i)-'0';
            for (int j = num2.length()-1;j>=0;j--){
                int n2 = num2.charAt(j)-'0';
                int sum = (res[i+j+1]+n1*n2);
                res[i+j+1] =sum%10;
                res[i+j] +=sum/10;
            }
        }
        StringBuilder result =new StringBuilder();
        for (int i =0;i<res.length;i++){
            if (i==0&&res[i]==0) continue;
            result.append(res[i]);
        }
        return result.toString();
    }
}
class Solution105AA{
    public TreeNode buildTree(int[] preorder, int[] inorder){
        int n = preorder.length;
        if (n==0) return null;
        int leftSize = indexOf(inorder,preorder[0]);//左子树大小
        int[] pre1 = Arrays.copyOfRange(preorder,1,1+leftSize);//左子树序列
        int[] pre2 = Arrays.copyOfRange(preorder,1+leftSize,n);//右子树序列
        int[] in1 =Arrays.copyOfRange(inorder,0,leftSize);//左子树
        int[] in2 = Arrays.copyOfRange(inorder,1+leftSize,n);//右子树

        TreeNode left = buildTree(pre1,in1);
        TreeNode right = buildTree(pre2,in2);
        return new TreeNode(preorder[0],left,right);


    }
    private int indexOf(int[] a,int x){//查找位置
        for (int i=0;;i++){
            if (a[i]==x){
                return i;
            }
        }
    }
}
class SolutionLCR140 {
    public ListNode trainingPlanA(ListNode head, int cnt) {
        List<Integer> vals = new ArrayList<>();
        ListNode cur = head;

        // Step1: 将链表的 val 存入数组
        while (cur != null) {
            vals.add(cur.val);
            cur = cur.next;
        }

        // Step2: 从 vals 的最后 cnt 位构造新链表
        ListNode dummy = new ListNode(0);
        ListNode p = dummy;
        for (int i = vals.size() - cnt; i < vals.size(); i++) {
            p.next = new ListNode(vals.get(i));
            p = p.next;
        }

        return dummy.next;
    }
    public ListNode trainingPlan(ListNode head, int cnt) {
        ListNode node1 = null;
        ListNode node2 = null;
        node1 = head;
        node2 = head;

        if (head.next==null){
            return head;
        }
        for (int i =1;i<cnt;i++){
            node2 = node2.next;
        }
        while (node2.next!=null){
            node1 = node1.next;
            node2 = node2.next;
        }
        return node1;
    }
}
class Solution151{
    public String reverseWords(String s){
        String[] str = s.trim().split(" ");
        StringBuilder res = new StringBuilder();
        for (int i =str.length-1;i>=0;i--){
            if (str[i].equals(""))continue;
            res.append(str[i]+" ");
        }
        return res.toString().trim();
    }
}
class Solution78A{
    private final List<List<Integer>> ans = new ArrayList<>();
    private final List<Integer> path = new ArrayList<>();
    private int[] nums;

    public List<List<Integer>> subsets(int[] nums){
        this.nums = nums;
        dfs(0);
        return ans;
    }
    private void dfs(int i ){
        if (i== nums.length){
            ans.add(new ArrayList<>(path));
            return;
        }
        dfs(i+1);
        path.add(nums[i]);
        dfs(i+1);
        path.remove(path.size()-1);
    }

}
class  Solution129{
    public int sumNumbers(TreeNode root) {
        return dfs(root, 0);
    }
    private int dfs(TreeNode node,int x){
        if (node==null){
            return 0;
        }
        x  =x*10+node.val;
        if (node.left==node.right){
            return x;
        }
        return dfs(node.left,x)+dfs(node.right,x);
    }
}
class MinStack{
    private Stack<Integer> stack;
    private Stack<Integer> min_stack;
    public MinStack(){
        stack = new Stack<>();
        min_stack = new Stack<>();
    }
    public     void push(int x){
        stack.push(x);
        if (min_stack.isEmpty()||x<=min_stack.peek()){
            min_stack.push(x);
        }
    }
    public     void pop(){
        if (stack.pop().equals(min_stack.peek())){
            min_stack.pop();
        }
    }
    public     int top(){
        return stack.peek();
    }
       public int getMin(){
        return min_stack.peek();
    }
}
class Solution101A{
    public boolean isSymmetric(TreeNode root){
        return root==null||recur(root.left,root.right);
    }
    private boolean recur(TreeNode L,TreeNode R){
        if (L==null&&R==null) return true;
        if (L==null||R==null||L.val!=R.val) return false;
        return recur(L.left,R.right)&&recur(R.left,L.right);
    }
}
class Solution34AA {
    public int[] searchRange(int[] nums, int target) {
        int start = lowBound(nums, target);
        if (start == nums.length || nums[start] != target) {
            return new int[]{-1, -1};
        }
        int end = lowBound(nums, target + 1) - 1;
        return new int[]{start, end};
    }

    private int lowBound(int[] nums, int target) {
        int left = 0, right = nums.length; // 注意 right = nums.length，开区间写法
        while (left < right) {
            int mid = (left + right) >>> 1;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid; // nums[mid] >= target
            }
        }
        return left;
    }
}
class Solution104AA{
    public int maxDepth(TreeNode root){
        if (root==null){
            return 0;
        }
        int left = maxDepth(root.left);
        int right = maxDepth(root.right);
        return Math.max(left,right)+1;
    }
}
class Solution93AA{
    void backtrack(List<Integer> state,int target,int[] choices,int start,List<List<Integer>> res){
        if (target==0){
            res.add(new ArrayList<>(state));
            return;
        }
        for (int i=start;i<choices.length;i++){
            if (target-choices[i]<0){
                break;
            }
            state.add(choices[i]);
            backtrack(state,target-choices[i],choices,i,res);
            state.remove(state.size()-1);
        }
    }
    public List<List<Integer>> combinationSum(int[] candidates,int target){
        List<Integer> state = new ArrayList<>();
        Arrays.sort(candidates);
        int start = 0;
        List<List<Integer>> res = new ArrayList<>();
        backtrack(state,target,candidates,start,res);
        return res;
    }
}
class Solution394{
    public String decodeString(String s) {
        Stack<Integer> countstack = new Stack<>();
        Stack<StringBuilder> stringstack = new Stack<>();
        StringBuilder currentString = new StringBuilder();
        int k = 0;

        for (char c:s.toCharArray()){
            if (Character.isDigit(c)){
                k = k*10+(c-'0');
            }
            else if (c=='['){
                countstack.push(k);
                k=0;
                stringstack.push(currentString);
                currentString = new StringBuilder();
            }
            else if (c==']'){
                int repeat = countstack.pop();
                StringBuilder sb = stringstack.pop();
                for (int i =0;i<repeat;i++){
                    sb.append(currentString);
                }
                currentString = sb;
            }else {
                currentString.append(c);
            }
        }
        return currentString.toString();
    }
}
class Solution144A{
    List<Integer> ans = new ArrayList<>();
    public List<Integer> preorderTraversal(TreeNode root){
        if (root==null){
            return new ArrayList<>();
        }
        ans.add(root.val);
        preorderTraversal(root.left);
        preorderTraversal(root.right);
        return ans;
    }
}
class Solution110A{
    public boolean isBalanced(TreeNode root){
        return getHeight(root) !=-1;
    }
    private int getHeight(TreeNode node){
        if (node==null){
            return 0;
        }
        int leftH = getHeight(node.left);
        if (leftH==-1){
            return -1;
        }
        int rightH = getHeight(node.right);
        if (rightH==-1||Math.abs(leftH-rightH)>1){
            return -1;
        }
        return Math.max(leftH,rightH)+1;
    }
}
class Solution64A{
    public int minPathSum(int[][] grid){
        for(int i=0;i<grid.length;i++){
            for (int j=0;j<grid[0].length;j++){
                if (i==0&&j==0) continue;
                else if (i==0) grid[i][j] = grid[i][j-1]+grid[i][j];
                else if (j==0) grid[i][j]=grid[i-1][j]+grid[i][j];
                else grid[i][j] += Math.min(grid[i - 1][j], grid[i][j - 1]);
            }
        }
        return grid[grid.length-1][grid[0].length-1];
    }
}
class Solution48A{
    public void rotate(int[][] matrix){
        int n  = matrix.length;
        int[][] tmp  = new int[n][];
        for (int i =0;i<n;i++){
            tmp[i] = matrix[i].clone();
        }
        for (int i =0;i<n;i++){
            for (int j = 0;j<n;j++){
                matrix[j][n-1-i] = tmp[i][j];
            }
        }
    }
}
class Solution221{
    int maximalSquare(char[][] matrix){
        int n  = matrix[0].length;
        int[] heights = new int[n+1];
        int ans = 0;
        for (char[] row :matrix){
            for (int j =0;j<n;j++){
                if (row[j]=='0'){
                    heights[j]=0;//柱子高度为0
                }else {
                    heights[j]++;
                }
            }
            ans = Math.max(ans,largesSize(heights));
        }
        return ans * ans;
    }
    private int largesSize(int[] heights){
        int n = heights.length;
        int[] st = new int[n];
        int top = -1;
        st[++top]  = -1;//在栈中只有一个数的时候，栈顶的「下面那个数」是 -1，对应 left[i] = -1 的情况
        int ans = 0;
        for (int right = 0;right<n;right++){
            int h = heights[right];
            while (top>0&&h<=heights[st[top]]){
                int i = st[top--];
                int left = st[top];
                ans =Math.max(ans,Math.min(heights[i],right-left-1));
            }
            st[++top] =right;
        }
        return ans;
    }
}
class Solution122A{
    public int maxProfit(int[] prices){
        int profit = 0;
        for (int i =1;i<prices.length;i++){
            int tmp = prices[i]-prices[i-1];
            if (tmp>0) profit+=tmp;
        }
        return profit;
    }
}
class Solution128A{
    public int longestConsecutive(int[] nums){
        int ans = 0;
        Set<Integer> st = new HashSet<>();
        for (int num:nums){
            st.add(num);
        }
        for (int x:st){
            if (st.contains(x-1)){
                continue;
            }
            int y = x+1;
            while (st.contains(y)){
                y++;
            }
            ans = Math.max(ans,y-x);
        }
        return ans;
    }
}
class Solution240{
    public boolean searchMatrix(int[][] matrix, int target){
        int i = 0;
        int j  = matrix[0].length-1;
        while (i<matrix.length&&j>=0){
            if (matrix[i][j]==target){
                return true;
            }
            if(matrix[i][j]<target){
                i++;
            }else {
                j--;
            }
        }
        return false;
    }
}
class Solution98AA {
    public boolean isValidBST(TreeNode root) {
        return dfs(root,Long.MIN_VALUE,Long.MAX_VALUE);

    }
    private boolean dfs(TreeNode node,long left,long right){
        if (node==null){
            return true;
        }
        long x = node.val;
        return x<right&&x>left&&dfs(node.left,left,x)&&dfs(node.right,x,right);
    }
}
class Solution234{
    public boolean isPalindrome(ListNode head) {
        if (head==null||head.next==null) return true;
        ListNode slow = head,fast = head;
        while (fast!=null&&fast.next!=null){
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode secondList = reverseList(slow);
        ListNode p1 = head;
        ListNode p2 = secondList;
        while (p2!=null){
            if (p1.val!= p2.val) return false;
            p1 = p1.next;
            p2 = p2.next;
        }
        return true;
    }
    private ListNode reverseList(ListNode node){
        ListNode pre = null,cur = node;
        while (cur!=null){
            ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
}
class Solution14AAA{
    public String longestCommonPrefix(String[] strs){
        if(strs.length==0) return "";
        String ans = strs[0];
        for (int i =1;i<strs.length;i++){
            int j =0;
            for (;j<ans.length()&&j<strs[i].length();j++){
                if(ans.charAt(j)!=strs[i].charAt(j)) break;
            }
            ans = ans.substring(0,j);
            if (ans.equals("")) return ans;
        }
        return ans;
    }
}
class Solution162{
    public int findPeakElement(int[] nums){
        int left = -1,right = nums.length-1;
        while (left+1<right){
            int mid = (left+right)>>>1;
            if (nums[mid]>nums[mid+1]){
                right = mid;
            }else {
                left = mid;
            }
        }
        return right;
    }
}
class Solution695{
    public int dfs(int[][] grid,int i,int j){
        if(i<0||j<0||i>=grid.length||j>=grid[0].length||grid[i][j]==0) return 0;
        int sum = 1;
        grid[i][j] = 0;
        sum +=dfs(grid,i+1,j);
        sum +=dfs(grid,i,j+1);
        sum +=dfs(grid,i-1,j);
        sum +=dfs(grid,i,j-1);
        return sum;
    }
    public int maxAreaOfIsland(int[][] grid){
        int max = 0;
        for (int i=0;i<grid.length;i++){
            for (int j=0;j<grid[0].length;j++){
                if (grid[i][j]==1){
                    max = Math.max(max,dfs(grid,i,j));
                }
            }
        }
        return max;
    }
}
class Solution113AA {
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        dfs(root,targetSum,path,ans);
        return ans;
    }
    private void dfs(TreeNode node, int left, List<Integer> path, List<List<Integer>> ans){
        if (node==null){
            return;
        }
        path.add(node.val);
        left -=node.val;
        if (node.left==node.right&&left==0){
            ans.add(new ArrayList<>(path));
        }else {
            dfs(node.left,left,path,ans);
            dfs(node.right,left,path,ans);
        }
        path.remove(path.size()-1);
    }
}
class Solution662{
    Map<Integer,Integer> map = new HashMap<>();
    int ans;
    public int widthOfBinaryTree(TreeNode root) {
       dfs(root,1,0);
       return ans;
    }
    void dfs(TreeNode root,int u,int depth){
        if (root==null) return;
        if (!map.containsKey(depth)) map.put(depth,u);
        ans = Math.max(ans,u-map.get(depth)+1);
        u = u-map.get(depth)+1;
        dfs(root.left,u<<1,depth+1);
        dfs(root.right,u<<1|1,depth+1);
    }
}
class Solution62A{
    public int uniquePaths(int m, int n){
        int[] f = new int[n+1];
        f[1]= 1;
        for (int i=0;i<m;i++){
            for (int j =0;j<n;j++){
                f[j+1] +=f[j];
            }
        }
        return f[n];
    }
}
class Solution152A{
    public int maxProduct(int[] nums) {
        int maxF = nums[0];
        int minF = nums[0];
        int ans = nums[0];

        for (int i=1;i<nums.length;i++){
            int cur = nums[i];
            int tmpMax = maxF,tmpMin = minF;
            maxF = Math.max(cur, Math.max(cur * tmpMax, cur * tmpMin));
            minF = Math.min(cur, Math.min(cur * tmpMax, cur * tmpMin));
            ans = Math.max(ans,maxF);
        }
        return ans;
    }
}
class Solution198{
    public int rob(int[] nums){
        int pre = 0,cur = 0,tmp;
        for(int num:nums){
            tmp = cur;
            cur = Math.max(pre+num,cur);
            pre = tmp;
        }
        return cur;
    }
}
class Solution179{
    public String largestNumber(int[] nums){
        String[] strs = new String[nums.length];
        for (int i=0;i<nums.length;i++){
            strs[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(strs,(x,y)->(y+x).compareTo(x+y));
        if (strs[0].equals("0")) return "0";
        StringBuilder res = new StringBuilder();
        for (String s:strs){
            res.append(s);
        }
        return res.toString();
    }
}
class Solution112A{
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if(root==null) return false;
        targetSum -=root.val;
        if (root.left==root.right&&root.left==null) return targetSum==0;
        return hasPathSum(root.left,targetSum)||hasPathSum(root.right,targetSum);
    }
}
class Solution560{
    public int subarraySum(int[] nums, int k) {
        int ans = 0;
        int s = 0;
        Map<Integer,Integer> cnt = new HashMap<>(nums.length);
        for(int x:nums){
            cnt.merge(s,1,Integer::sum);
            s +=x;
            ans +=cnt.getOrDefault(s-k,0);
        }
        return ans;
    }
}
class Solution227{
    static Stack<Integer> num = new Stack<Integer>();
    static Stack<Character> op = new Stack<Character>();
    static HashMap<Character, Integer> map = new HashMap<Character, Integer>();
    static void eval(){
        int b = num.pop();
        int a = num.pop();
        char c =op.pop();
        int r = 0;
        if (c=='+') r = a+b;
        else if (c=='-') r = a-b;
        else if (c=='*') r = a*b;
        else r = a/b;
        num.push(r);
    }
    public int calculate(String s) {
        s = '0'+s;
        map.put('+',1);
        map.put('-', 1);
        map.put('*', 2);
        map.put('/', 2);
        for (int i =0;i<s.length();i++){
            char c = s.charAt(i);
            if (c==' ')continue;
            if (c>='0'&&c<='9'){
                int x = 0;
                while (i<s.length()&&s.charAt(i)>='0'&&s.charAt(i)<=9){
                    x =x*10+s.charAt(i++)-'0';
                }
                i--;
                num.push(x);
            }else {
                while(!op.isEmpty() && map.get(op.peek()) >= map.get(c)) eval();
                op.push(c);
            }
        }
        while (!op.isEmpty()) eval();
        return num.pop();
     }
}
 class Solution227A {
    public int calculate(String s) {
        int num = 0;
        char sign = '+'; // 初始默认是加法
        Stack<Integer> stack = new Stack<>();
        int n = s.length();

        for (int i = 0; i < n; i++) {
            char c = s.charAt(i);

            // 如果是数字，累加
            if (Character.isDigit(c)) {
                num = num * 10 + (c - '0');
            }

            // 如果是运算符，或者是最后一个字符，就处理当前数字
            if ((!Character.isDigit(c) && c != ' ') || i == n - 1) {
                if (sign == '+') {
                    stack.push(num);
                } else if (sign == '-') {
                    stack.push(-num);
                } else if (sign == '*') {
                    stack.push(stack.pop() * num);
                } else if (sign == '/') {
                    stack.push(stack.pop() / num); // 题目要求整数除法
                }
                sign = c;
                num = 0;
            }
        }

        // 把栈里所有数字加起来
        int result = 0;
        for (int val : stack) {
            result += val;
        }
        return result;
    }
}
class Solution169 {
    public int majorityElement(int[] nums) {
        Map<Integer,Integer> cnt = new HashMap<>();
        for (int x:nums){
            cnt.put(x, cnt.getOrDefault(x, 0) + 1);
            if (cnt.get(x)>nums.length/2){
              return x;
            }
        }
        return -1;
    }
}
class Solution169A{
    public int majorityElement(int[] nums){
        int x = 0,votes = 0;
        for (int num:nums){
            if (votes==0) x = num;
            votes +=num ==x?1:-1;
        }
        return x;
    }
}
class Solution226{
    public TreeNode invertTree(TreeNode root) {
        if(root==null) return null;
        TreeNode tmp =root.left;
        root.left = invertTree(root.right);
        root.right = invertTree(tmp);
        return root;
    }
}
class Solution718{
    public int findLength(int[] nums1, int[] nums2){
        int n  = nums1.length;
        int m = nums2.length;
        int ans = 0;
        int [][] f= new int[n+1][m+1];
        for(int i =0;i<n;i++){
            for (int j=0;j<m;j++){
                if(nums1[i]==nums2[j]){
                    f[i+1][j+1] = f[i][j]+1;
                    ans = Math.max(ans,f[i+1][j+1]);
                }
            }
        }
        return ans;
    }
}
class Solution209{
    public int minSubArrayLen(int target, int[] nums){
        int n =  nums.length;
        int left =0,ans = Integer.MAX_VALUE;
        int sum = 0;
        for (int right  = 0;right<n;right++){
            sum +=nums[right];
            while (sum>=target){
                ans = Math.min(ans,right-left+1);
                sum -=nums[left++];
            }
        }
        return ans == Integer.MAX_VALUE ? 0 : ans;
    }
}
class Solution139{
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> set = new HashSet<>(wordDict);
        int n = s.length();
        boolean[] dp =new boolean[n+1];
        dp[0] = true;
        for (int i=1;i<=n;i++){
            for (int j=0;j<i;j++){
                if (dp[j]&&set.contains(s.substring(j,i))){
                    dp[i] =true;
                    break;
                }
            }
        }
        return dp[n];
    }
}
class Solution83A{
    public ListNode deleteDuplicates(ListNode head) {
        if (head==null) return null;
        ListNode cur = head;
        while (cur!=null&&cur.next!=null){
            if (cur.val==cur.next.val){
                cur.next = cur.next.next;
            }else {
                cur =cur.next;
            }
        }
        return head;
    }
}
class Solution24A{
    public ListNode swapPairs(ListNode head) {
        if (head==null||head.next==null) return head;
        ListNode tmp = head.next;
        head.next = swapPairs(tmp.next);
        tmp.next = head;
        return tmp;
    }
}
class Solution283A{
    public void moveZeroes(int[] nums){
        int slow = 0;
        for (int fast = 0;fast<nums.length;fast++){
            if (nums[fast]!=0){
                int temp = nums[fast];
                nums[fast] = nums[slow];
                nums[slow++] = temp;
            }
        }
    }
}
class Solution468{
    public String validIPAddress(String queryIP){
        boolean flag = false;
        for (int i=0;i<queryIP.length();i++){
            if (queryIP.charAt(i)=='.'){
                flag = true;
                break;
            }else if (queryIP.charAt(i)==':'){
                flag  = false;
                break;
            }
        }
        if (!flag){
            String [] split = queryIP.split(":",-1);
            if (split.length!=8){
                return "Neither";
            }
            for (String s:split){
                if (s.length()>4||s.isEmpty()) return "Neither";
                for (char c:s.toCharArray()){
                    if (!String.valueOf(c).matches("[0-9a-fA-F]")){
                        return "Neither";
                    }
                }
            }
            return "IPv6";
        }else {
            String[] split = queryIP.split("\\.",-1);
            if (split.length!=4) return "Neither";
            for (String s:split){
                if (s.isEmpty()) return "Neither";
                try {
                    int num = Integer.parseInt(s);
                    if (num>255||!String.valueOf(num).equals(s)){
                        return "Neither";
                    }
                }catch (NumberFormatException e){
                    return "Neither";
                }
            }
            return "IPv4";
        }
    }
}
class Solution739A {
    public int[] dailyTemperatures(int[] temperatures) {
        int n =temperatures.length;
        int[] res = new int[n];
        Deque<Integer> stack = new ArrayDeque<>();
        for (int i=0;i<n;i++){
            while (!stack.isEmpty()&&temperatures[i]>temperatures[stack.peek()]){
                int pre = stack.pop();
                res[pre]  =i-pre;
            }
            stack.push(i);
        }
        return res;
    }
}
class Solution138A{
    public Node copyRandomList(Node head){
        if (head==null) return null;
        for(Node cur = head;cur!=null;cur = cur.next.next){
            cur.next = new Node(cur.val,cur.next);
        }
        for (Node cur =head;cur!=null;cur = cur.next.next){
            if (cur.random!=null){
                cur.next.random = cur.random.next;
            }
        }
        Node newHead = head.next;
        Node cur = head;
        for (;cur.next.next!=null;cur=cur.next){
            Node copy = cur.next;
            cur.next = copy.next;
            copy.next =copy.next.next;
        }
        cur.next = null;
        return newHead;

    }
}
class Solution224A{
    public int calculate(String s){
            int res = 0;
            int num = 0;
            int sign = 1; // 当前符号，1 表示正，-1 表示负
            Deque<Integer> stack = new LinkedList<>();
            for(int i=0;i<s.length();i++){
                char ch = s.charAt(i);
                if (Character.isDigit(ch)){
                    num = num*10+(ch-'0');
                }else if (ch=='+'){
                    res  +=sign*num;
                    num = 0;
                    sign = 1;
                }else if (ch=='-'){
                    res +=sign*num;
                    num = 0;
                    sign=-1;
                } else if (ch=='('){
                    stack.push(res);
                    stack.push(sign);
                    res= 0;
                    sign =1;
                }else if (ch==')'){
                    res +=sign*num;
                    num = 0;
                    int preSign = stack.pop();
                    int preRes = stack.pop();
                    res = preRes+preSign*res;
                }
            }
            res +=sign*num;
            return res;
    }
}
class Solution153{
    public int findMin(int[] nums){
        int left = 0,right = nums.length-1;
        if (nums[left]<nums[right]) return nums[0];
        while (left<right){
            int mid = (left+right)>>>1;
            if (nums[mid]>nums[right]){
                left = mid+1;
            }else {
                right = mid;
            }
        }
        return nums[left];
    }
}
class Solution207{
    public boolean canFinish(int numCourses, int[][] prerequisites){
        List<Integer>[] g  = new ArrayList[numCourses];
        Arrays.setAll(g,i->new ArrayList<>());;
        for (int[] p : prerequisites) {
            g[p[1]].add(p[0]);
        }

        int[] colors =new int[numCourses];
        for (int i=0;i<numCourses;i++){
            if (colors[i]==0&&dfs(i,g,colors)){
                return false;
            }

        }
        return true;
    }
    private boolean dfs(int x, List<Integer>[] g, int[] colors){
        colors[x]=1;
        for (int y:g[x]){
            if (colors[y]==1||colors[y]==0&&dfs(y,g,colors)){
                return true;
            }
        }
        colors[x]=2;
        return false;
    }
}
class Solution79A{
    public boolean exist(char[][] board, String word) {
        int r = board.length;
        int t = board[0].length;
        boolean[][] visited = new boolean[r][t];
        for (int i=0;i<r;i++) {
            for (int j = 0; j < t; j++) {
                if (board[i][j] == word.charAt(0)) {
                    if (dfs(board, word, i, j, 0, visited)) {
                        return true;
                    }

                }
            }
        }
        return false;
    }
    private boolean dfs(char[][] board, String word, int row, int col, int index, boolean[][] visited){
        if (index==word.length()){
            return true;
        }
        if (row < 0 || row >= board.length || col < 0 || col >= board[0].length ||
                board[row][col] != word.charAt(index) || visited[row][col]) {
            return false;
        }
        visited[row][col]=true;
        boolean found = dfs(board, word, row + 1, col, index + 1, visited) ||
                dfs(board, word, row - 1, col, index + 1, visited) ||
                dfs(board, word, row, col + 1, index + 1, visited) ||
                dfs(board, word, row, col - 1, index + 1, visited);
        visited[row][col]= false;
        return found;
    }
}
class Solution47AA {
    List<Integer> nums;
    List<List<Integer>> res;

    void swap(int a, int b) {
        int tmp = nums.get(a);
        nums.set(a, nums.get(b));
        nums.set(b, tmp);
    }

    void dfs(int x) {
        if (x == nums.size() - 1) {
            res.add(new ArrayList<>(nums));
            return;
        }
        HashSet<Integer> set = new HashSet<>();
        for (int i = x; i < nums.size(); i++) {//固定x
            if (set.contains(nums.get(i))) {
                continue;
            }
            set.add(nums.get(i));
            swap(i, x);
            dfs(x + 1);
            swap(x, i);
        }
    }

    public List<List<Integer>> permuteUnique(int[] nums) {
        this.res = new ArrayList<>();
        this.nums = new ArrayList<>();
        for (int num : nums) {
            this.nums.add(num);
        }
        dfs(0);
        return res;
    }
}
class Solution402A {
    public String removeKdigits(String num, int k) {
        StringBuilder sb = new StringBuilder();
        for(char ch:num.toCharArray()){
            while (!sb.isEmpty()&&ch<sb.charAt(sb.length()-1)&&k>0){
                sb.deleteCharAt(sb.length()-1);
                k--;
            }
            sb.append(ch);
        }
        while (!sb.isEmpty()&&k>0){
            sb.deleteCharAt(sb.length()-1);
            k--;
        }
        while (!sb.isEmpty()&&sb.charAt(0)=='0'){
            sb.deleteCharAt(0);
        }
        return sb.isEmpty()?"0":sb.toString();

    }
}
class Solution76A{
    public String minWindow(String S, String t) {
        int[] cnt = new int[128];
        int less = 0;
        for (char c:t.toCharArray()){
            if (cnt[c]==0){
                less++;
            }
            cnt[c]++;
        }
        char[] s = S.toCharArray();
        int m  = s.length;
        int ansleft  = -1;
        int ansRight = m;

        int left =0;
        for (int right = 0;right<m;right++){
            char c =s[right];
            cnt[c]--;
            if (cnt[c]==0){
                less--;
            }
            while (less==0){
                if (right-left<ansRight-ansleft){
                    ansleft = left;
                    ansRight = right;
                }
                char x = s[left];
                if (cnt[x]==0){
                    less++;
                }
                cnt[x]++;
                left++;
            }
        }
        return ansleft<0?"":S.substring(ansleft,ansRight+1);
    }
}
class Solution124A {
    private int ans = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        dfs(root);
        return ans;
    }
    private int dfs(TreeNode root){
        if (root==null){
            return 0;
        }
        int lval = dfs(root.left);
        int rval  =dfs(root.right);
        ans  = Math.max(ans,lval+rval+root.val);
        return Math.max(Math.max(lval,rval)+ root.val,0);
    }
}
class Trie{
    private static class Node{
        Node[] son = new Node[26];
        boolean end;
    }
    private final Node root = new Node();
    public void insert(String word) {
        Node cur = root;
        for (char c:word.toCharArray()){
            c -='a';
            if (cur.son[c]==null){
                cur.son[c] = new Node();
            }
            cur = cur.son[c];
        }
        cur.end = true;
    }
    public boolean search(String word) {
        return find(word)==2;
    }
    public boolean startsWith(String prefix) {
        return find(prefix)!=0;
    }
    private int find(String word){
        Node cur = root;
        for (char c:word.toCharArray()){
            c-='a';
            if(cur.son[c]==null){
                return 0;
            }
            cur = cur.son[c];
        }
        return cur.end?2:1;
    }
}