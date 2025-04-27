import java.util.Stack;

public class Souluntion {
    public boolean isValid(String s) {
        // 如果字符串长度是奇数，直接返回false
        if (s.length() % 2 == 1) {
            return false;
        }

        Stack<Character> stack = new Stack<>();

        // 遍历字符串中的每个字符
        for (char ch : s.toCharArray()) {
            // 如果是左括号，入栈
            if (ch == '(' || ch == '[' || ch == '{') {
                stack.push(ch);
            } else if (ch == ')') {
                // 如果栈非空且栈顶是相应的左括号，弹栈
                if (!stack.isEmpty() && stack.peek() == '(') {
                    stack.pop();
                } else {
                    return false;
                }
            } else if (ch == ']') {
                if (!stack.isEmpty() && stack.peek() == '[') {
                    stack.pop();
                } else {
                    return false;
                }
            } else if (ch == '}') {
                if (!stack.isEmpty() && stack.peek() == '{') {
                    stack.pop();
                } else {
                    return false;
                }
            }
        }
        // 如果栈为空，说明所有括号匹配，返回true，否则false
        return stack.isEmpty();
    }
}
