import java.text.SimpleDateFormat;
import java.util.*;
import java.util.stream.Collectors;

public class Test2 {

    public static void main(String[] args) {
        int a = 0;
        try {
            a = 1/0;
        } catch (Exception e) {
            System.out.println("error");
        }
        System.out.println("success");
        System.out.println(a);
    }

}

class Person {
    long a;

    public long getA() {
        return a;
    }
}
