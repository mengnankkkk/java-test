public class UnionFind {
    private int[] parent;
    private int[] rank;

    public UnionFind(int size) {
        parent = new int[size];
        rank = new int[size];
        for (int i=0;i<size;i++){
            parent[i] = i;
            rank[i] =1;                                                                  
        }
    }
    public int find(int p){
        if (parent[p]!=p){
            parent[p] = find(parent[p]);
        }
        return parent[p];
    }
    public void union(int p,int q){
        int rootP = find(p);
        int rootQ = find(q);
        if (rootQ!=rootP){
            if (rank[rootP]>rank[rootQ]){
                parent[rootQ] = rootP;
            }else if (rank[rootP]<rank[rootQ]){
                parent[rootP] = rootQ;
            }else {
                parent[rootQ] = rootP;
                rank[rootP]++;
            }
        }
    }
    public boolean connected(int p,int q){
        return find(p) ==find(q);
    }

    public static void main(String[] args) {
        UnionFind unionFind = new UnionFind(10);
        unionFind.union(1,2);
        unionFind.union(2,3);
        System.out.println(unionFind.connected(1, 3));
        System.out.println(unionFind.connected(1, 4));

    }
}
