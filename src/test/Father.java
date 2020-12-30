package test;

public class Father {
    protected int tag;
    protected int setTag(){
        return -1;
    }
    public void print(){
        tag = setTag();
        System.out.println(tag);
        print1();
    }

    public void print1(){
        System.out.println("1");
    }

    public static void main(String[] args){
        Son1 son1 = new Son1();
        //Father father = new Father();
        son1.print();
    }
}
