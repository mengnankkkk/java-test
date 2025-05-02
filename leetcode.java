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




