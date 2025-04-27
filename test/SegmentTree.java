public class SegmentTree {
    private int tree[];
    private int data[];

    public SegmentTree(int[] nums) {
        if (nums.length > 0) {
            int n = nums.length;
            data = nums;
            tree = new int[4 * n];
            buildTree(0, 0, n - 1);
        }
    }

    private void buildTree(int node, int start, int end) {
        if (start == end) {
            tree[node] = data[start];
        } else {
            int mid = (start + end) / 2;
            int leftChild = 2 * node + 1;
            int rightChild = 2 * node + 2;
            buildTree(leftChild, start, mid);
            buildTree(rightChild, mid + 1, end);
            tree[node] = tree[leftChild] + tree[rightChild];
        }
    }

    public int query(int L, int R) {
        return queryRec(0, 0, data.length - 1, L, R);
    }

    private int queryRec(int node, int start, int end, int L, int R) {
        if (R < start || L > end) {
            return 0;
        }
        if (L <= start && end <= R) {
            return tree[node];
        }
        int mid = (start + end) / 2;
        int leftChild = 2 * node + 1;
        int rightChild = 2 * node + 2;
        int leftSum = queryRec(leftChild, start, mid, L, R);
        int rightSum = queryRec(rightChild, mid + 1, end, L, R);
        return leftSum + rightSum;
    }
    public void update(int index,int val){
        updateRec(0,0,data.length-1,index,val);
    }
    private void updateRec(int node,int start,int end,int index,int val) {
        if (start == end) {
            data[index] = val;
            tree[node] = val;
        } else {
            int mid = (start + end) / 2;
            int leftChild = 2 * node + 1;
            int rightChild = 2 * node + 2;
            if (index <= mid) {
                // 更新左子树
                updateRec(leftChild, start, mid, index, val);
            } else {
                // 更新右子树
                updateRec(rightChild, mid + 1, end, index, val);
            }
            tree[node] = tree[leftChild] + tree[rightChild];
        }
    }
}
