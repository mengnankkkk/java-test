public class testannotation {
    private int[] data;
    private int size;
    private int capacity;
    
    public testannotation(int capacity) {
        this.capacity = capacity;
        this.data = new int[capacity];
        this.size = 0;
    }

    public boolean insert(int element) {
        if (size >= capacity) {
            System.out.println("顺序表已满，无法插入新元素。");
            return false;
        }


        int i = 0;
        while (i < size && data[i] < element) {
            i++;
        }


        if (i < size && data[i] == element) {
            System.out.println("元素 " + element + " 已存在，不插入。");
            return false;
        }


        for (int j = size - 1; j >= i; j--) {
            data[j + 1] = data[j];
        }

        // 插入元素
        data[i] = element;
        size++;
        System.out.println("插入元素 " + element + " 成功。");
        return true;
    }


    public boolean delete(int element) {

        int i = 0;
        while (i < size && data[i] != element) {
            i++;
        }


        if (i == size) {
            System.out.println("元素 " + element + " 不存在，无法删除。");
            return false;
        }


        for (int j = i; j < size - 1; j++) {
            data[j] = data[j + 1];
        }
        size--;
        System.out.println("删除元素 " + element + " 成功。");
        return true;
    }


    public void display() {
        System.out.print("顺序表内容: ");
        for (int i = 0; i < size; i++) {
            System.out.print(data[i] + " ");
        }
        System.out.println();
    }

    public static void main(String[] args) {
        testannotation orderedList = new testannotation(10);

        orderedList.insert(10);
        orderedList.insert(5);
        orderedList.insert(20);
        orderedList.insert(15);

        orderedList.display();

        orderedList.insert(10);

        orderedList.delete(5);
        orderedList.display();

        orderedList.delete(30);
        orderedList.display();
    }
}
