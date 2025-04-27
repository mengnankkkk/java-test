class Stackbase<I extends Number> {
    private int[] arr;
    private int top;
    private int maxSize;
    public Stackbase(int size){
        maxSize = size;
        arr = new int[maxSize];
        top = -1;
    }
    public void push(int value){
        arr[++top] = value;
    }
    public int pop() {
        return arr[top--];
    }
    public boolean isEmpty(){
        return top  == -1;
    }
}
public class DecimalToBase {
    public static void convert(int n,int d){
        if (n==0){
            System.out.println(0);
            return;
        }
        Stackbase<Number> stackbase = new Stackbase<Number>(32);
        while (n>0){
            stackbase.push(n%d);
            n /=d;
        }
        while (!stackbase.isEmpty()){
            System.out.print(stackbase.pop());
        }
        System.out.println();
    }

    public static void main(String[] args) {
        convert(1348,8);
        convert(988,12);

    }
}
