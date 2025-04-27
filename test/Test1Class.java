package javahight;

import org.junit.Test;

import java.io.ObjectStreamException;

public class Test1Class{
    @Test
    public void save() throws Exception {
        Students s1 = new Students("zhou","18");
        ObjectSaver.saveObject(s1);
    }
}