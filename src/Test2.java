import java.text.SimpleDateFormat;
import java.util.*;

public class Test2 {

    public static void main(String[] args){
        /*ArrayList<String> arrayList = new ArrayList<>();
        arrayList.add("a");
        arrayList.add("b");
        arrayList.add("c");
        arrayList.add("d");
        //arrayList.add("e");
        //arrayList.add("f");

        for(int i=0;i<arrayList.size();i++){
            if(arrayList.size()>5){
                System.out.println(i);
                continue;
            }
            System.out.println("lalala");
            System.out.println(i);
        }*/
        /*long x= 1L;
        if(x>0){
            System.out.println("yes");
        }else {
            System.out.println("no");
        }

        System.out.println("con");*/
        //List o = null;
        //System.out.println(o.size());
        //test1();


        /*long a = 8197078931L;
        String string = ""+a;
        System.out.println(string);

        Map<String,String> map = new HashMap<>();
        System.out.println(map.get("999"));*/


        // String string = "260616:0";
        /*
        if(string.startsWith("260616")){
            String trash = string.substring(string.indexOf(":")+1);
            if(trash.equals("0")){
                System.out.println(trash);
            }

        }


        // System.out.println(string.substring(0));
        // System.out.println(string);
        ArrayList<String> arrayList = new ArrayList<>();
        arrayList.add("1");
        arrayList.add("2");
        arrayList.add("3");
        System.out.println(arrayList);

        String pre = "qwertfdsdvsv";
        System.out.println();

        System.out.println(pre.indexOf('8'));


        Map<String, Map<String, String>> map = new HashMap<>();
        Map<String, String> map1 = new HashMap<>();

        map.put("map1", map1);

        map1.put("map1", "val1");

        System.out.println(map.get("map2").get("map2"));
        */

        // String str = "{\"result\":\"success\",\"message\":\"成功！\"}";

        Person person = new Person();
        System.out.println(person.age);

        Map<Integer, String> map = new HashMap<>();
        map.put(1,"1");
        System.out.println(map);




    /*Date date=new Date();
    date.setTime(1599809747159L);
    SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    System.out.println(simpleDateFormat.format(date));*/
    }


    public static void test1(){
        int a = 1;
        a = test2(a);
        System.out.println(a);
    }

    public static int test2(int n){
        n = n+1;
        return n;
    }

}

class Person{
    String result;
    String message;
    int age;
}
