public class Stack<T> {
    private T[] stackArray;
    private int maxSize;
    private int top;

    public Stack(){
        int size = 0;
        this.maxSize =size;
        this.stackArray = (T[]) new Object[maxSize];
        this.top= -1;
    }
    public void push(T value){
        if (isFull()){
            throw new StackOverflowError("full!");
        }
        stackArray[++top] = value;
    }
    public T pop(){
        if (isEmpty()){
            throw new IllegalStateException("empty!");
        }
        return stackArray[top--];
    }
    public T peek(){
        if (isEmpty()){
            throw new IllegalStateException("empty!");

        }
        return stackArray[top];
    }
    public boolean isEmpty() {
        return top == -1;  // 如果栈顶指针为 -1，则栈为空
    }
    public boolean isFull() {
        return top == maxSize - 1;
    }
    public int size(){
        return top  + 1;
    }
}
