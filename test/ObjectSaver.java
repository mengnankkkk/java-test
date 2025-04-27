package javahight;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.lang.reflect.Field;

public class ObjectSaver { // 修改类名避免与 java.lang.Object 冲突
    public static void saveObject(Object obj) throws Exception {
        // 设置文件路径，确保路径有效
        PrintStream ps = new PrintStream(new FileOutputStream("output.txt", true));
        Class c = obj.getClass();//反射第一步
        Field[] fields = c.getDeclaredFields();//获取成员变量
        String cName = c.getSimpleName();//获取简短名称
        ps.println("------" + cName + "-------");//输出-----Students----
        for (Field field : fields) {
            String name = field.getName();//获取名字
            field.setAccessible(true); // 修正为 field.setAccessible(true)//遍历成员变量
            String value = field.get(obj) + "";//获取值
            ps.println(name + "=" + value);//输出
        }
        ps.close(); // 关闭 PrintStream
    }
}