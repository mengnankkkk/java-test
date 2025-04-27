 class RedBlackTree {
    private static final boolean RED = true;
    private static final boolean BLACK  = false;
    private class Node{
        int key;
        Node left,right,parent;
        boolean color;
        Node(int key){
            this.key = key;
            this.color = RED;
        }
    }
    private Node root;
    private void leftRotate(Node x){//左旋
        Node y =x.right;
        x.right =y.left;
        if (y.left!=null){
            y.left.parent=x;
        }
        y.parent = x.parent;
        if (x.parent==null){
            root=y;
        }else if (x==x.parent.left){
            x.parent.left = y;
        }else {
            x.parent.right = y;
        }
        y.left = x;
        x.parent = y;
    }

}
