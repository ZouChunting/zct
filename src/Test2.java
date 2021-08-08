import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.stream.Collectors;

public class Test2 {

    public static void main(String[] args) throws ParseException {
        Calendar cal = Calendar.getInstance();
        TimeZone timeZone = TimeZone.getTimeZone("America/Chicago");
        cal.setTimeZone(timeZone);
        cal.set(Calendar.YEAR, 2020);
        cal.set(Calendar.MONTH, 1);
        cal.set(Calendar.DAY_OF_MONTH, 29);
        cal.set(Calendar.HOUR_OF_DAY, 0);
        cal.set(Calendar.MINUTE, 0);
        cal.set(Calendar.SECOND, 0);
        cal.set(Calendar.MILLISECOND, 0);
        System.out.println(cal.getTime());
        System.out.println(cal.getTimeInMillis());
    }

}

class Person {
    long a;

    public long getA() {
        return a;
    }
}
