package leetcode;

import javafx.util.Pair;

import java.util.*;

public class DP {
    // 70. 爬楼梯
    public int climbStairs(int n) {
        if (n < 2) {
            return n;
        }
        int res = 0;
        int pre0 = 1;
        int pre1 = 1;
        for (int i = 2; i < n + 1; i++) {
            res = pre0 + pre1;
            pre0 = pre1;
            pre1 = res;
        }
        return res;
    }

    // 198. 打家劫舍
    public int rob(int[] nums) {
        int res = 0;
        if (nums.length == 0) {
            return res;
        }

        int pre0 = 0;
        int pre1 = 0;
        for (int num : nums) {
            res = Math.max(num + pre0, pre1);
            pre0 = pre1;
            pre1 = res;
        }

        return res;
    }

    // 413. 等差数列划分
    public int numberOfArithmeticSlices(int[] nums) {
        int res = 0;
        if (nums.length < 3) {
            return res;
        }
        int pre = 0;
        int cur = 0;
        for (int i = 2; i < nums.length; i++) {
            if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) {
                cur = pre + 1;
            } else {
                cur = 0;
            }
            pre = cur;
            res += cur;
        }

        return res;
    }

    // 64. 最小路径和
    public int minPathSum(int[][] grid) {
        if (grid.length == 0 || grid[0].length == 0) {
            return 0;
        }
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (i - 1 >= 0 && j - 1 >= 0) {
                    grid[i][j] += Math.min(grid[i - 1][j], grid[i][j - 1]);
                } else if (i - 1 < 0 && j - 1 >= 0) {
                    grid[i][j] += grid[i][j - 1];
                } else if (i - 1 >= 0) {
                    grid[i][j] += grid[i - 1][j];
                }
            }
        }
        return grid[grid.length - 1][grid[0].length - 1];
    }

    // 542. 01 矩阵
    public int[][] updateMatrix(int[][] matrix) {
        int row = matrix.length;
        int col = matrix[0].length;

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (matrix[i][j] == 1) {
                    // 必须将值预先置为最大
                    matrix[i][j] = row + col;
                    if (i > 0) {
                        matrix[i][j] = Math.min(matrix[i][j], matrix[i - 1][j] + 1);
                    }
                    if (j > 0) {
                        matrix[i][j] = Math.min(matrix[i][j], matrix[i][j - 1] + 1);
                    }
                }
            }
        }

        for (int i = row - 1; i >= 0; i--) {
            for (int j = col - 1; j >= 0; j--) {
                if (matrix[i][j] != 0) {
                    if (i < row - 1) {
                        matrix[i][j] = Math.min(matrix[i][j], matrix[i + 1][j] + 1);
                    }
                    if (j < col - 1) {
                        matrix[i][j] = Math.min(matrix[i][j], matrix[i][j + 1] + 1);
                    }
                }
            }
        }
        return matrix;
    }

    // 221. 最大正方形
    public int maximalSquare(char[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }
        int res = 0;
        int[][] dp = new int[matrix.length + 1][matrix[0].length + 1];
        for (int i = 1; i <= matrix.length; i++) {
            for (int j = 1; j <= matrix[0].length; j++) {
                if (matrix[i - 1][j - 1] == '1') {
                    dp[i][j] = 1 + Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1]));
                    res = Math.max(res, dp[i][j]);
                }
            }
        }
        return res * res;
    }

    // 279. 完全平方数
    // 数学渣渣真的心累
    public int numSquares(int n) {
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            dp[i] = i;
            for (int j = 1; j * j <= i; j++) {
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
            }
        }
        return dp[n];
    }

    // 91. 解码方法
    public int numDecodings(String s) {
        if (s.length() == 0) {
            return 0;
        }
        char[] nums = s.toCharArray();
        if (nums[0] == '0') {
            return 0;
        }
        int pre_1 = 1;
        int pre_2 = 1;
        int res = 1;

        for (int i = 1; i < nums.length; i++) {
            int num = nums[i] - '0';
            int pre = nums[i - 1] - '0';
            if (num == 0) {
                // 自己不行，自己都不行的情况只可能是0
                if (pre == 0 || pre > 2) {
                    // 跟前面的组合也不行，那就是废了
                    return 0;
                } else {
                    // 自己不行，但是前面的可以带，也就是说，自己必须跟前者绑定，此时只有能由i-2来变换组合，所以i=i-2
                    res = pre_2;
                }
            } else {
                // 自己行
                if (pre == 1 || (pre == 2 && num < 7)) {
                    // 不仅自己行，而且跟前面的绑定起来也行，此时i-1和i-2不管谁变换组合都可以，所以i=(i-1)+(i-2)
                    res = pre_1 + pre_2;
                } else {
                    // 自己行，但是不能跟前面的绑定，此时i=i-1
                    res = pre_1;
                }
            }
            pre_2 = pre_1;
            pre_1 = res;
        }
        return res;
    }

    // 139. 单词拆分
    public boolean wordBreak(String s, List<String> wordDict) {
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i < dp.length; i++) {
            for (int j = 0; j < i; j++) {
                // 在此坐标i下，是否能在满足前面阶段单词都存在于列表中的同时，最新截断也存在于列表中
                if (dp[j] && wordDict.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }

        return dp[s.length()];
    }

    // 300. 最长递增子序列
    public int lengthOfLIS(int[] nums) {
        if (nums.length < 2) {
            return nums.length;
        }
        int res = 0;
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        dp[0] = 1;
        int max = dp[0];
        for (int i = 1; i < nums.length; i++) {
            for (int j = i - 1; j >= 0; j--) {
                if (nums[j] < nums[i]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            res = Math.max(res, dp[i]);
        }

        return res;
    }

    // 上题的优化解法，贪心+二分查找
    public int lengthOfLIS2(int[] nums) {
        if (nums.length < 2) {
            return nums.length;
        }
        int[] tail = new int[nums.length];
        tail[0] = nums[0];
        int end = 0;
        for (int num : nums) {
            if (num > tail[end]) {
                end++;
                tail[end] = num;
            } else {
                // 如果num值比队列最后一个数小，就向前二分查找，找到比num值大的最小数
                int left = 0;
                int right = end;
                while (left < right) {
                    int mid = (left + right) / 2;
                    if (tail[mid] < num) {
                        left = mid + 1;
                    } else {
                        right = mid;
                    }
                }
                tail[left] = num;
            }
        }
        return end + 1;
    }

    // 1143. 最长公共子序列
    // 我是智障
    public int longestCommonSubsequence(String text1, String text2) {
        int[][] dp = new int[text1.length() + 1][text2.length() + 1];
        for (int i = 1; i < text1.length() + 1; i++) {
            for (int j = 1; j < text2.length() + 1; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    // 如果两字符相同，说明此时已遍历过的字符串在原来的基础上可以喜加1
                    // 例："字符串1"+"字符1"与"字符串2"+"字符2"
                    // "字符1"和"字符2"相同，则考虑在"字符串1"和"字符串2"最佳重合的基础上加1，
                    // 即"字符串1"+"字符1"与"字符串2"+"字符2"的最佳重合
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    // 如果两字符不同，说明不能喜加1，则退而求其次
                    // 例："字符串1"+"字符1"与"字符串2"+"字符2"
                    // "字符1"和"字符2"不相同，则退而考虑"字符串1"+"字符1"、"字符串2"和"字符串1"、"字符串2"+"字符2"的最佳重合
                    // 取两者中的较大值，即"字符串1"+"字符1"与"字符串2"+"字符2"的最佳重合
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[text1.length()][text2.length()];
    }

    // 背包问题
    // 0-1背包
    // 各物品重量-各物品价值-物品数量-背包容量
    public int knapsack(int[] weights, int[] values, int N, int W) {
        int[][] dp = new int[N + 1][W + 1];
        for (int i = 1; i <= N; i++) {
            // i代表物品编号
            for (int j = 1; j <= W; j++) {
                // j代表背包容量
                if (j < weights[i - 1]) {
                    // 此时的背包容量小于当前物品的重量，因此该物品不能放进去
                    // 所以此时的最佳选择仍是遍历上一物品时的选择
                    dp[i][j] = dp[i - 1][j];
                } else {
                    // 此时背包容量大于当前物品的重量，该物品可以放进去
                    // 所以此时考虑，不放置该物品的最大价值和放置该物品的最大价值
                    // 不放置该物品的最大价值即上一状态的最佳选择
                    // 放置该物品的最大价值即该物品价值，与除去该物品重量的剩余重量可承载的最佳选择
                    dp[i][j] = Math.max(dp[i - 1][j], values[i - 1] + dp[i - 1][j - weights[i - 1]]);
                }
            }
        }
        return dp[N][W];
    }

    // 对背包问题进行空间优化
    // 这个压缩空间的方法很秒，最好自己画数组模拟一下计算，不然比较难理解
    // 完全体的数组，每次甚至都不用更新全。
    public int knapsack2(int[] weights, int[] values, int N, int W) {
        int[] dp = new int[W + 1];
        for (int i = 1; i <= N; i++) {
            // 注意此处j的遍历范围
            for (int j = W; j >= weights[i - 1]; j--) {
                // 动态规划基于状态转移思想，即，将当前状态的最优解，寄托于前一状态的最优解解决。
                // 背包问题的前置状态有：1.物品--；负重-- 2.物品-- 3.负重--
                // 思考此题，背包问题的当前状态与那种前置状态相关呢？
                // 背包问题与负重强相关，随着负重的增加调整物品的负载，即用不同的物品去适应负重
                // 查看上例，dp[i][j]的值只依赖于i-1的状态，即负重变化的状态
                // 即只与列相关，与行无关，因此可以去除多余空间
                dp[j] = Math.max(dp[j], values[i - 1] + dp[j - weights[i - 1]]);
            }
        }
        return dp[W];
    }

    // 完全背包
    // 背包问题自己动手画二维数组图更有助于理解
    public int knapsack3(int[] weights, int[] values, int N, int W) {
        int[][] dp = new int[N + 1][W + 1];
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= W; j++) {
                if (j < weights[i - 1]) {
                    dp[i][j] = dp[i - 1][j];
                } else {
                    // 此处是和0-1背包问题不同的地方 dp[i][j - weights[i - 1]]
                    // 对于同一物品在递增的容量的变量来说，实际上每当容量变化时，前面状态的容量时刻，实际上已经被考虑过，物品叠加的可能
                    // 仍然符合状态转移
                    dp[i][j] = Math.max(dp[i - 1][j], values[i - 1] + dp[i][j - weights[i - 1]]);
                }
            }
        }
        return dp[N][W];
    }

    // 完全背包问题的空间优化
    public int knapsack4(int[] weights, int[] values, int N, int W) {
        int[] dp = new int[W + 1];
        for (int i = 1; i <= N; i++) {
            // 注意此处j的遍历范围，是正向遍历
            // 结合前文0-1背包问题，是逆向遍历，从最大容量慢慢往低容量计算，那种情况是没办法使用到多个物品的
            for (int j = weights[i - 1]; j <= W; j++) {
                dp[j] = Math.max(dp[j], values[i - 1] + dp[j - weights[i - 1]]);
            }
        }
        return dp[W];
    }

    // 背包问题其实难度不大，只是比较抽象，多动手画图，不要图快，慢慢理解吧

    // 416. 分割等和子集
    public boolean canPartition(int[] nums) {
        int sum = Arrays.stream(nums).sum();
        if (sum % 2 != 0) {
            return false;
        }
        int target = sum / 2;
        int[] dp = new int[target + 1];
        for (int i = 1; i <= nums.length; i++) {
            for (int j = target; j >= nums[i - 1]; j--) {
                dp[j] = Math.max(dp[j], nums[i - 1] + dp[j - nums[i - 1]]);
                if (dp[j] == target) {
                    return true;
                }
            }
        }

        return false;
    }

    // 474. 一和零
    // 注意：这题是背包问题的三维版了
    // m个0 n个1
    public int findMaxForm(String[] strs, int m, int n) {
        int[][] dp = new int[m + 1][n + 1];
        for (String str : strs) {
            int[] count = countStr(str);
            int count0 = count[0];
            int count1 = count[1];
            for (int i = m; i >= count0; i--) {
                for (int j = n; j >= count1; j--) {
                    dp[i][j] = Math.max(dp[i][j], dp[i - count0][j - count1] + 1);
                }
            }
        }

        return dp[m][n];
    }

    int[] countStr(String str) {
        int[] count = new int[2];

        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == '0') {
                count[0]++;
            } else if (str.charAt(i) == '1') {
                count[1]++;
            }
        }

        return count;
    }

    // 322. 零钱兑换
    // 这就是传说中的完全背包
    // 理解时变换下思维
    // 重要：理解动态规划还是要自己多画数组。本体可以从1 2 5和5 2 1两个顺序来画
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 2);
        dp[0] = 0;

        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (coin <= i) {
                    dp[i] = Math.min(dp[i], 1 + dp[i - coin]);
                }
            }
        }

        return dp[amount] == amount + 2 ? -1 : dp[amount];
    }

    // 72. 编辑距离
    // 朴素的经验总结是不够的，还需要有合理的公式说明
    public int minDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                if (i == 0) {
                    dp[i][j] = j;
                } else if (j == 0) {
                    dp[i][j] = i;
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j - 1] + (word1.charAt(i - 1) == word2.charAt(j - 1) ? 0 : 1), Math.min(dp[i - 1][j], dp[i][j - 1]) + 1);
                }
            }
        }
        return dp[m][n];
    }

    // 650. 只有两个键的键盘
    // 此题在教程中有更优的边界，但说实话我理解起来有点困难
    // 此题的官方题解也很妙
    public int minSteps(int n) {
        int[] dp = new int[n];
        dp[0] = 0;
        for (int i = 1; i < n; i++) {
            dp[i] = i + 1;
            for (int j = 2; j <= (i + 1) / 2; j++) {
                if ((i + 1) % j == 0) {
                    dp[i] = Math.min(dp[i], dp[i / j] + j);
                }
            }
        }
        return dp[n - 1];
    }

    // 如此理解，一个数注定可以由有1*n得到
    // 假设这个数注定能由x*x构成，那么剩下的乘积组合，必定一个在1和x之间，另一个在x和n之间
    // 所以有Math.sqrt(n)作为边界
    // 举例，6*6=36，6又可以由2,3继续分解
    // 所以dp[36]=dp[6】+dp[6]=dp[2]+dp[18]=dp[3]+dp[12]=dp[4]+dp[9]
    // 最终dp[36]=dp[2]+dp[2]+dp[3]+dp[3]
    // 这题又可以理解为一道分解因数的题
    // 另外，如何朴素地理解dp[36]=dp[6]+dp[6]呢？
    // 即：第一个dp[6]是得到6个A的步骤，再把这6个A看成一个整体称为A`，第二个dp[6]即为得到6个A`的步骤
    // 如此递推下去，以同样的思想理解dp[6]=dp[2]+dp[3]，达到状态转移
    // 再想不明白还可以倒过来想
    // dp[36]=dp[18]+dp[2]=dp[6]+dp[3]+dp[2]=dp[3]+dp[2]+dp[3]+dp[2]
    public int minSteps2(int n) {
        int[] dp = new int[n];
        dp[0] = 0;
        for (int i = 1; i < n; i++) {
            dp[i] = i + 1;
            int m = (int) Math.sqrt(i + 1);
            for (int j = 2; j <= m; j++) {
                if ((i + 1) % j == 0) {
                    dp[i] = Math.min(dp[i], dp[i / j] + dp[j - 1]);
                    break;
                }
            }
        }
        return dp[n - 1];
    }

    // 这个解就更妙了
    // 由上一题我们可以知道此题基于分解因数去解决
    // 另外我们可以想见，一个数n如果是素数，那么就必然只能以一步步来，所以结果是
    // 基于这两种结论，我们可以将n逐步分解为素数
    // 假设n=36，按以下代码分解步骤如下：36->(2+)18->(2+2+)9->(2+2+3+)3->(2+2+3+3)
    // 假设n=23，则：23
    // 比较上面2个例子，23遍历了23次，36则遍历了4次
    // 46的话，则需要遍历23+1次
    public int minSteps3(int n) {
        int res = 0;
        int step = 2;

        while (n > 1) {
            while (n % step == 0) {
                res += step;
                n /= step;
            }
            step++;
        }

        return res;
    }

    // 10. 正则表达式匹配
    public boolean isMatch(String s, String p) {
        int lenS = s.length();
        int lenP = p.length();
        boolean[][] dp = new boolean[lenS + 1][lenP + 1];
        dp[0][0] = true;
        for (int i = 0; i <= lenS; i++) {
            for (int j = 1; j <= lenP; j++) {
                if (i == 0) {
                    if (p.charAt(j - 1) == '*') {
                        dp[i][j] = dp[i][j - 2];
                    }
                } else {
                    char sc = s.charAt(i - 1);
                    char pc = p.charAt(j - 1);
                    if (pc == sc || pc == '.') {
                        dp[i][j] = dp[i - 1][j - 1];
                    }
                    if (pc == '*') {
                        char pre = p.charAt(j - 2);
                        if (pre == '.' || pre == sc) {
                            // 字符能与目标串匹配，能选择的情况包括
                            // 1.保留一个，则取各自退一步的值
                            // 2.不保留，取当前串前一步的值
                            // 3.不保留甚至要去掉前一个，取当前串前两步的值
                            // 4.多复制一个，取目标串前一步值
                            dp[i][j] = dp[i - 1][j - 1] || dp[i][j - 1] || dp[i][j - 2] || dp[i - 1][j];
                        } else {
                            // 字符与目标无法匹配，则只能利用*的特性，把这个字符消掉，即字符串的前两位与目标串的匹配情况
                            dp[i][j] = dp[i][j - 2];
                        }
                    }
                }
            }
        }

        return dp[lenS][lenP];
    }

    // 121. 买卖股票的最佳时机
    public int maxProfit(int[] prices) {
        int len = prices.length;
        int max = 0;
        int res = 0;
        for (int i = len - 1; i >= 0; i--) {
            max = Math.max(max, prices[i]);
            res = Math.max(res, max - prices[i]);
        }
        return res;
    }

    // 188. 买卖股票的最佳时机 IV
    // 关于这道题的解法可以思考几个问题
    // 根据题意可以确定，每次只能拥有一支股票，即买了之后必须跟着卖掉的操作，即在再次购买之前，手上的股票已经全部卖掉了，此时为空
    // 所以不需要担心重复购买了同一支股票
    // 假设天数小于k*2，比如说prices的长度是1，而k是2，即最多可以买卖两次
    // 在进行dp运算时，相当于对这一支股票进行了两次重复的买进有卖掉操作。
    // 相信当k等于1时，不难理解
    // 当k等于2时，第一次买入卖出的操作逻辑跟k=1时是一样的
    // 第二次的买入卖出逻辑可以这样理解
    // 不论下文代码设定，这里假设当前状态为dp[i][j]。处于第二次买卖状态
    // dp[i-1][j-1]即前一天股票买卖结束的状态，当天这支股票可以买入
    // dp[i-1][j]即前一天股票买卖结束，并且又重新购入了（并不一定就是前一天的那支股票），此时这只股票不可以买，必须等卖掉才能有买入操作。
    // 简言之，就是到了这天这支股票操作还是不操作的选择，操作会怎样、不操作会怎样，因此对于dp[i][j]的计算，只关注i-1的状态
    // 或者倒推来想，到最后一天了，这个股票我是卖还是不卖呢？
    public int maxProfit(int k, int[] prices) {
        int m = prices.length;
        int n = 2 * k + 1;
        int[][] dp = new int[m][n];
        for (int j = 1; j < n - 1; j += 2) {
            dp[0][j] = -prices[0];
        }
        // dp[i][1]代表第i天买入股票的状态
        for (int i = 1; i < m; i++) {
            // 注意j += 2，此时j代表状态买入，j+1代表状态卖出
            for (int j = 1; j < n - 1; j += 2) {
                // dp[i - 1][j]代表取上一状态，即当天不买
                // dp[i - 1][j - 1] - prices[i]
                dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - 1] - prices[i]);
                // dp[i - 1][j + 1]代表当天不卖
                // dp[i - 1][j] + prices[i]代表当天卖掉
                dp[i][j + 1] = Math.max(dp[i - 1][j + 1], dp[i - 1][j] + prices[i]);
            }
        }

        return dp[m - 1][n - 1];
    }

    // 309. 最佳买卖股票时机含冷冻期
    // 这道题理解起来其实稍微有点歧义，比如买入或卖掉股票之后能不能超过2天不操作？？？
    public int maxProfit2(int[] prices) {
        int len = prices.length;
        if (len < 2) {
            return 0;
        }
        int[][] dp = new int[len][3];
        dp[0][1] = -prices[0];
        // 0.不操作状态（即冷冻状态）1.买入状态 2.卖出状态
        // 每一行代表当天的股票的状态
        for (int i = 1; i < len; i++) {
            // 如果当天不操作，则前一天可以是任何状态
            // 这里实际计算下来只可能是不操作或者卖掉，因为不操作本身取的就是前一天的最大值，而买入是消耗性操作，不可能比其他两个大
            dp[i][0] = Math.max(dp[i - 1][0], Math.max(dp[i - 1][1], dp[i - 1][2]));
            // 如果当天想买入，则前一天只能是不操作状态或者保留前一天的买入状态
            dp[i][1] = Math.max(dp[i - 1][0] - prices[i], dp[i - 1][1]);
            // 如果当天想卖掉，则前一天可以是不操作（其实是把当天的买了又卖掉）和买入状态
            dp[i][2] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i]);
        }
        return Math.max(dp[len - 1][0], Math.max(dp[len - 1][1], dp[len - 1][2]));
    }

    // 213. 打家劫舍 II
    public int rob2(int[] nums) {
        int len = nums.length;
        if (len == 0) {
            return 0;
        }
        if (len == 1) {
            return nums[0];
        }
        if (len == 2) {
            return Math.max(nums[0], nums[1]);
        }
        // 分偷第一个和不偷第一个两种情况
        int[] rob = new int[4];
        // 1.偷了第一个
        rob[0] = nums[0];
        rob[1] = nums[0];
        // 2.没偷第一个
        rob[2] = nums[1];
        rob[3] = 0;
        int tmp0, tmp1;
        // 不算到最后一个
        for (int i = 2; i < len - 1; i++) {
            // 1.偷了第一个
            tmp0 = nums[i] + rob[1]; // 此时要偷
            tmp1 = Math.max(rob[0], rob[1]); // 此时不偷
            rob[0] = tmp0;
            rob[1] = tmp1;
            // 2.没偷第一个
            tmp0 = nums[i] + rob[3]; // 此时要偷
            tmp1 = Math.max(rob[2], rob[3]); // 此时不偷
            rob[2] = tmp0;
            rob[3] = tmp1;
        }
        // 没有偷第一个的情况补算一下最后一个
        tmp0 = nums[len - 1] + rob[3];
        tmp1 = Math.max(rob[2], rob[3]);
        rob[2] = tmp0;
        rob[3] = tmp1;

        return Math.max(Math.max(rob[0], rob[1]), Math.max(rob[2], rob[3]));
    }

    // 53. 最大子序和
    public int maxSubArray(int[] nums) {
        int len = nums.length;
        if (len == 0) {
            return 0;
        }
        int res = nums[0], tmp = nums[0];
        for (int i = 1; i < len; i++) {
            // 要么从当前的i开始，要么接上前面一起
            tmp = Math.max(nums[i], nums[i] + tmp);
            res = Math.max(res, tmp);
        }
        return res;
    }

    // 343. 整数拆分
    public int integerBreak(int n) {
        int[] dp = new int[n + 1];
        if (n < 2) {
            return dp[n];
        }
        dp[2] = 1;
        for (int i = 3; i <= n; i++) {
            for (int j = 1; j <= i / 2; j++) {
                dp[i] = Math.max(dp[i], Math.max(j, dp[j]) * Math.max(i - j, dp[i - j]));
            }
        }
        return dp[n];
    }

    // 583. 两个字符串的删除操作
    public int minDistance2(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 1; i < m + 1; i++) {
            for (int j = 1; j < n + 1; j++) {
                dp[i][j] = word1.charAt(i - 1) == word2.charAt(j - 1) ? dp[i - 1][j - 1] + 1 : Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
        return m + n - 2 * dp[m][n];
    }

    // 646. 最长数对链
    // 用了贪心
    public int findLongestChain(int[][] pairs) {
        if (pairs.length == 0 || pairs[0].length == 0) {
            return 0;
        }
        // 按照每个数对的第二个数排序
        Arrays.sort(pairs, Comparator.comparingInt(p -> p[1]));
        int res = 1;
        int init = pairs[0][1];
        for (int i = 1; i < pairs.length; i++) {
            if (pairs[i][0] > init) {
                res++;
                init = pairs[i][1];
            }
        }
        return res;
    }

    // 376. 摆动序列
    public int wiggleMaxLength(int[] nums) {
        if (nums.length == 0 || nums.length == 1) {
            return nums.length;
        }
        int up = 1;
        int down = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > nums[i - 1]) {
                up = down + 1;
            } else if (nums[i] < nums[i - 1]) {
                down = up + 1;
            }
        }

        return Math.max(up, down);
    }

    // 494. 目标和
    public int findTargetSumWays(int[] nums, int S) {
        int len = nums.length;
        if (len == 0) {
            return 0;
        }
        if (len == 1) {
            if (S == nums[0] || S == -nums[0]) {
                return 1;
            } else {
                return 0;
            }
        }

        Map<Integer, Integer> dp = new HashMap<>();
        dp.put(nums[0], 1);
        dp.put(-nums[0], dp.getOrDefault(-nums[0], 0) + 1);
        for (int i = 1; i < len; i++) {
            Map<Integer, Integer> cur = new HashMap<>();
            for (int target : dp.keySet()) {
                cur.put(target - nums[i], cur.getOrDefault(target - nums[i], 0) + dp.get(target));
                cur.put(target + nums[i], cur.getOrDefault(target + nums[i], 0) + dp.get(target));
            }
            dp.clear();
            dp.putAll(cur);
        }

        return dp.getOrDefault(S, 0);
    }

    // 494. 目标和
    // 此题解需要通过数学理解，转化为正常的0-1背包问题
    public int findTargetSumWays2(int[] nums, int S) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (S > sum || (S + sum) % 2 == 1) {
            return 0;
        }
        int target = (S + sum) / 2;
        int[] dp = new int[target + 1];
        dp[0] = 1; // 这里的意思是，目标值0总有保底办法达成，那就是什么都不装
        for (int num : nums) {
            for (int j = target; j >= num; j--) {
                dp[j] = dp[j] + dp[j - num];
            }
        }
        return dp[target];
    }

    // 714. 买卖股票的最佳时机含手续费
    public int maxProfit(int[] prices, int fee) {
        if (prices.length == 0) {
            return 0;
        }
        // 默认在买的时候扣除手续费
        // 状态：买入、卖出、无操作（是整体状态，而不是某一天的状态）
        int[] status = new int[3];
        status[0] = -prices[0] - fee;
        for (int i = 1; i < prices.length; i++) {
            // 不操作状态
            status[2] = Math.max(status[1], status[2]);
            // 卖出状态
            status[1] = Math.max(status[1], prices[i] + status[0]);
            // 买入状态
            status[0] = Math.max(status[0], -prices[i] - fee + status[2]);
        }
        return Math.max(status[1], status[2]);
    }

    public static void main(String[] args) {
        DP dp = new DP();

        int n = 18;
        // System.out.println(dp.climbStairs(n));

        int[] nums = new int[]{7, 9, 3, 8, 0, 2, 4, 8, 3, 9};
        // System.out.println(dp.numberOfArithmeticSlices(nums));

        int[][] grid = new int[][]{
                {1, 3, 1},
                {1, 5, 1},
                {4, 2, 1}
        };
        // System.out.println(dp.minPathSum(grid));

        int[][] matrix = new int[][]{
                {1, 0, 1, 1, 0, 0, 1, 0, 0, 1},
                {0, 1, 1, 0, 1, 0, 1, 0, 1, 1},
                {0, 0, 1, 0, 1, 0, 0, 1, 0, 0},
                {1, 0, 1, 0, 1, 1, 1, 1, 1, 1},
                {0, 1, 0, 1, 1, 0, 0, 0, 0, 1},
                {0, 0, 1, 0, 1, 1, 1, 0, 1, 0},
                {0, 1, 0, 1, 0, 1, 0, 0, 1, 1},
                {1, 0, 0, 0, 1, 1, 1, 1, 0, 1},
                {1, 1, 1, 1, 1, 1, 1, 0, 1, 0},
                {1, 1, 1, 1, 0, 1, 0, 0, 1, 1}
        };
        // System.out.println(Arrays.deepToString(dp.updateMatrix(matrix)));

        // System.out.println(dp.numSquares(n));

        // String s = "leetcode";
        // System.out.println(dp.numDecodings(s));

        List<String> wordDict = new ArrayList<>();
        wordDict.add("leet");
        wordDict.add("code");
        // System.out.println(dp.wordBreak(s, wordDict));

        // System.out.println(dp.lengthOfLIS(nums));

        // System.out.println(dp.canPartition(nums));

        String word1 = "zoologicoarchaeologist";
        String word2 = "zoogeologist";
        // System.out.println(dp.minDistance(word1, word2));

        // System.out.println(dp.minSteps3(36));

        String s = "aaa";
        String p = ".*";
        // System.out.println(dp.isMatch(s, p));

        int[] prices = new int[]{0, 0};
        // System.out.println(dp.maxProfit2(prices));

        // System.out.println(dp.maxSubArray(nums));

        // System.out.println(dp.integerBreak(9));

        // System.out.println(dp.wiggleMaxLength(nums));

        System.out.println(dp.findTargetSumWays2(nums, 0));
    }
}
/*
{1, 0, 1, 1, 0, 0, 1, 0, 0, 1},
                {0, 1, 1, 0, 1, 0, 1, 0, 1, 1},
                {0, 0, 1, 0, 1, 0, 0, 1, 0, 0},
                {1, 0, 1, 0, 1, 1, 1, 1, 1, 1},
                {0, 1, 0, 1, 1, 0, 0, 0, 0, 1},
                {0, 0, 1, 0, 1, 1, 1, 0, 1, 0},
                {0, 1, 0, 1, 0, 1, 0, 0, 1, 1},
                {1, 0, 0, 0, 1, 1, 1, 1, 0, 1},
                {1, 1, 1, 1, 1, 1, 1, 0, 1, 0},
                {1, 1, 1, 1, 0, 1, 0, 0, 1, 1}
{0, 0, 0},
                {0, 1, 0},
                {1, 1, 1}
 */
