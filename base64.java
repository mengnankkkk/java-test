import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.Scanner;

public class base64 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        String originalText = scanner.nextLine();

        // 将字符串转换为UTF-8编码的字节数组
        byte[] encodedData = originalText.getBytes(StandardCharsets.UTF_8);

        // 使用Base64对字节数组进行编码
        String encodeText = Base64.getEncoder().encodeToString(encodedData);

        // 输出Base64编码后的字符串
        System.out.println(encodeText);  // 输出: "SGVsbG8gV29ybGQh"
    }
}
