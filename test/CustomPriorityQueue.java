import java.util.PriorityQueue;

// 任务类，包含任务名、优先级、执行时间
class Task implements Comparable<Task> {
    private String name;
    private int priority;       // 优先级，值越小优先级越高
    private long executionTime;  // 执行时间（时间戳）

    // 构造函数
    public Task(String name, int priority, long executionTime) {
        this.name = name;
        this.priority = priority;
        this.executionTime = executionTime;
    }

    // 实现 compareTo 方法，首先比较优先级，如果优先级相同，则比较执行时间
    @Override
    public int compareTo(Task other) {
        if (this.priority != other.priority) {
            return Integer.compare(this.priority, other.priority);  // 优先级小的先处理
        } else {
            return Long.compare(this.executionTime, other.executionTime);  // 时间早的先处理
        }
    }

    @Override
    public String toString() {
        return "Task{name='" + name + "', priority=" + priority + ", executionTime=" + executionTime + '}';
    }
}

public class CustomPriorityQueue {
    public static void main(String[] args) {
        // 创建优先队列，并将 Task 任务对象放入队列
        PriorityQueue<Task> taskQueue = new PriorityQueue<>();

        // 添加任务，传入任务名称、优先级和执行时间
        taskQueue.add(new Task("任务1", 3, System.currentTimeMillis() + 1000));
        taskQueue.add(new Task("任务2", 1, System.currentTimeMillis()));
        taskQueue.add(new Task("任务3", 2, System.currentTimeMillis() + 500));
        taskQueue.add(new Task("任务4", 2, System.currentTimeMillis() + 200));

        // 依次处理优先队列中的任务
        while (!taskQueue.isEmpty()) {
            System.out.println("处理任务: " + taskQueue.poll());
        }
    }
}
