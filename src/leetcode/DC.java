package leetcode;

import java.util.ArrayList;
import java.util.List;

public class DC {
    // 241. 为运算表达式设计优先级
    public List<Integer> diffWaysToCompute(String expression) {
        return dfs(expression, 0, expression.length() - 1);
    }

    // 画树理解
    List<Integer> dfs(String expression, int left, int right) {
        int index = left;
        int num = expression.charAt(index) - '0';
        List<Integer> res = new ArrayList<>();
        // 组装运算符左侧的数字
        while (index < right && Character.isDigit(expression.charAt(index + 1))) {
            index++;
            num = num * 10 + (expression.charAt(index) - '0');
        }
        if (index == right) {
            res.add(num);
            return res;
        }
        for (int i = index + 1; i <= right; i++) {
            if (Character.isDigit(expression.charAt(i))) {
                // 实际上应该不会跑到这一步，因为前面都规避掉了
                continue;
            }
            List<Integer> leftNums = dfs(expression, left, i - 1);
            List<Integer> rightNums = dfs(expression, i + 1, right);
            for (int leftNum : leftNums) {
                for (int rightNum : rightNums) {
                    char opt = expression.charAt(i);
                    if (opt == '+') {
                        res.add(leftNum + rightNum);
                    } else if (opt == '-') {
                        res.add(leftNum - rightNum);
                    } else if (opt == '*') {
                        res.add(leftNum * rightNum);
                    }
                }
            }
        }

        return res;
    }

    // 932. 漂亮数组
    public void beautifulArray(int N) {

    }

    public static void main(String[] args) {
        DC dc = new DC();

        String expression = "11";
        System.out.println(dc.diffWaysToCompute(expression));
    }
}
