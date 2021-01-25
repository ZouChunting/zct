package leetcode;

public class Shuzu {
    // 867. 转置矩阵
    public int[][] transpose(int[][] A) {
        // 行
        int row = A.length;
        // 列
        int col = A[0].length;
        int[][] res = new int[col][row];
        for(int i=0;i<col;i++){
            for(int j=0;j<row;j++){
                res[i][j] = A[j][i];
            }
        }
        return res;
    }


    public static void main(String[] args){
        Shuzu shuzu = new Shuzu();
        int[][] A = {{1,2,3}, {4,5,6}};
        System.out.println(shuzu.transpose(A).toString());
    }
}
