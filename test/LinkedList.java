public class LinkedList<T> {
    private static class Node<T>{
        T data;
        Node<T> prev;
        Node<T> next;

        Node(T data){
            this.data = data;
        }
    }
    private Node<T> head;
    private Node<T> tail;
    private int size;

    public LinkedList() {
        this.head = null;
        this.tail = null;
        this.size = 0;
    }
    public void addFirst(T data){
        Node<T> newNode = new Node<>(data);
        if (head == null) {
            head = newNode;
            tail = newNode;
        } else {
            newNode.next = head;
            head.prev = newNode;
            head = newNode;
        }
        size++;
    }
    public T removeFirst(){
        if (head ==null){
            throw new IllegalStateException("有错误");
        }
        Node<T> removedNode = head;
        if (head ==tail){
            head = null;
            tail = null;
        }else {
            head = head.next;
            head.prev = null;
        }
        size--;
        return removedNode.data;
    }
    public T removeLast(){
        if (head ==null){
            throw new IllegalStateException("有错误");
        }
        Node<T> removedNode = head;
        if (head==tail){
            head = null;
            tail = null;
        }else {
            tail = tail.prev;
            tail.next = null;
        }
        size--;
        return removedNode.data;
    }
    public boolean contains(T data){
        Node<T> current = head;
        while (current != null){
            if (current.data.equals(data)){
                return true;
            }
            current = current.next;
        }
        return false;
    }
    public boolean remove(T data){
        Node<T> current = head;
        while (current != null){
            if (current.data.equals(data)){
                if (current == head){
                    removeFirst();
                } else if (current == tail) {
                    removeLast();
                }else {
                    current.prev.next = current.next;
                    current.next.prev = current.prev;
                    size--;
                }
                return true;
            }
            current = current.next;
        }
        return false;
    }
    public void reverse(){
        Node<T> current = head;
        Node<T> temp = null;
        while (current != null){
            temp = current.prev;
            current.prev = current.next;
            current.next = temp;
            current = current.prev;
        }
        if (temp!=null){
            head = temp.prev;
        }
    }
    public int getSize(){
        return size;
    }
    public boolean isEmpty(){
        return size ==0;
    }
    public void printForward(){
        Node<T> current = head;
        while (current != null){
            System.out.println(current.data+" ");
            current = current.next;
        }
        System.out.println();
    }
    public void printBackward(){
        Node<T> current = tail;
        while (tail!=null){
            System.out.println(current.data+" ");
            current = current.prev;
        }
        System.out.println();
    }
}