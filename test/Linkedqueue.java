public class Linkedqueue {
    private Node front;
    private Node rear;
    private int size;

    public Linkedqueue(Node front, Node rear, int size) {
        this.front = front;
        this.rear = rear;
        this.size = size;
    }

    public boolean isEmpty() {
        return front == null;
    }

    public void enqueue(int date) {
        Node newNode = new Node(date);
        if (rear == null) {
            front = rear = newNode;
        } else {
            rear.next = newNode;
            rear = newNode;
        }
        size++;
        System.out.println("入队元素是" + date);
    }

    public int dequeue() {
        if (isEmpty()) {
            System.out.println("队列为空");
            return -1;
        }
        int data = front.data;
        front = front.next;

        if (front == null) {
            rear = null;
        }
        size--;
        System.out.println("出队元素是" + data);
        return data;
    }

    public int peek() {
        if (isEmpty()) {
            System.out.println("队列为空");
            return -1;
        }
        return front.data;
    }
    public int getSize() {
        return size;
    }
}
