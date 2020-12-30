package test;

public class Son1 extends Father {
    @Override
    protected int setTag(){
        return 2;
    }

    public void print1(){
        System.out.println("11");
    }
}
