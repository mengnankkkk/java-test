import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class reg_exp {
    public static String render(String template, Map<String, String> values) {
        // 定义匹配 ${key} 的正则表达式
        Pattern pattern = Pattern.compile("\\$\\{(\\w+)}");
        Matcher matcher = pattern.matcher(template);

        // 使用 StringBuffer 来高效拼接替换后的字符串
        StringBuffer result = new StringBuffer();

        while (matcher.find()) {
            String key = matcher.group(1);  // 获取 ${key} 中的 key
            String replacement = values.getOrDefault(key, "");  // 从 Map 中获取 key 对应的值
            matcher.appendReplacement(result, replacement);  // 替换 ${key} 为对应的值
        }

        matcher.appendTail(result);  // 添加剩余的部分
        return result.toString();
    }

    public static void main(String[] args) {
        // 定义模板
        String template = "Hello, ${name}! You are learning ${lang}!";

        // 定义替换内容的 Map
        Map<String, String> values = Map.of(
                "name", "Bob",
                "lang", "Java"
        );

        // 渲染模板并输出结果
        String output = render(template, values);
        System.out.println(output);  // 输出：Hello, Bob! You are learning Java!
    }
}
