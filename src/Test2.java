import java.text.SimpleDateFormat;
import java.util.*;

public class Test2 {

    public static void main(String[] args) {
        String s = "0123456";
        System.out.println(s.substring(0,2));  // 左闭右开

        System.out.println(6/2*3);
    }


    public static void test1() {
        int a = 1;
        a = test2(a);
        System.out.println(a);
    }

    public static int test2(int n) {
        n = n + 1;
        return n;
    }

}

class Person {
    String result;
    String message;
    int age;
}
