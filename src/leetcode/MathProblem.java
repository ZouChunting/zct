package leetcode;

import sun.nio.cs.ext.MacArabic;

import java.util.Arrays;
import java.util.Random;

public class MathProblem {
    // 数学问题

    Random random = new Random();

    // 求公因数
    public int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    // 求公倍数
    public int lcm(int a, int b) {
        return a * b / gcd(a, b);
    }

    // 204. 计数质数
    public int countPrimes(int n) {
        boolean[] flags = new boolean[n];
        for (int i = 2; i * i < n; i++) {
            // 该数为质数
            if (!flags[i]) {
                for (int j = i * i; j < n; j += i) {
                    // 将其倍数标记为非质数
                    flags[j] = true;
                }
            }
        }
        int count = 0;
        for (int i = 2; i < n; i++) {
            if (!flags[i]) {
                count++;
            }
        }

        return count;
    }

    // 计数质数 优化
    public int countPrimes2(int n) {
        if (n <= 2) {
            return 0;
        }
        boolean[] flags = new boolean[n];
        int count = n;
        for (int i = 2; i * i < n; i++) {
            // 该数为质数
            if (!flags[i]) {
                for (int j = i * i; j < n; j += i) {
                    // 将其倍数标记为非质数
                    if (!flags[j]) {
                        flags[j] = true;
                        count--;
                    }
                }
            }
        }

        return count - 2;
    }

    // 504. 七进制数
    /*
    输入: 100
    输出: "202"
     */
    public String convertToBase7(int num) {
        if (num == 0) {
            return "0";
        }
        boolean flag = false;
        if (num < 0) {
            num = -num;
            flag = true;
        }
        String res = "";
        while (num != 0) {
            int tmp = num % 7;
            res = tmp + res;
            num /= 7;
        }
        return flag ? "-" + res : res;
    }

    // 172. 阶乘后的零
    // 2*5得0，5的数量比2多
    public int trailingZeroes(int n) {
        int res = 0;
        while (n >= 5) {
            res += (n /= 5);
        }
        return res;
    }

    // 415. 字符串相加
    public String addStrings(String num1, String num2) {
        StringBuilder res = new StringBuilder();
        int i = num1.length() - 1;
        int j = num2.length() - 1;
        int remainder = 0;
        while (i >= 0 || j >= 0 || remainder > 0) {
            if (i >= 0) {
                remainder += num1.charAt(i) - '0';
                i--;
            }
            if (j >= 0) {
                remainder += num2.charAt(j) - '0';
                j--;
            }
            res.append(remainder % 10);
            remainder = remainder / 10;
        }

        return res.reverse().toString();
    }

    // 326. 3的幂
    public boolean isPowerOfThree(int n) {
        return n > 0 && 1162261467 % n == 0;
    }

    public boolean isPowerOfThree1(int n) {
        if (n < 1) {
            return false;
        }
        while (n % 3 == 0) {
            n /= 3;
        }
        return n == 1;
    }

    public boolean isPowerOfThree2(int n) {
        // 这里应该用log10，这种方法会出现精度问题，但是我没看懂
        return n > 0 && (Math.log(n) / Math.log(3)) % 1 == 0;
    }

    // 随机与取样

    // 这个二分查找再认真看看
    public int bSearch(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = (left + right) / 2;
            if (target < nums[mid]) {
                right = mid;
            } else if (target > nums[mid]) {
                left = mid + 1;
            } else {
                return mid;
            }
        }
        return right;
    }

    // 168. Excel表列名称
    // A:1 Z:26 AA:27
    // 1-26进制
    public String convertToTitle(int columnNumber) {
        StringBuilder res = new StringBuilder();
        while (columnNumber > 0) {
            int tmp = columnNumber % 26;
            if (tmp == 0) {
                res.append('Z');
                columnNumber -= 26;
            } else {
                tmp -= 1;
                res.append((char) ('A' + tmp));
            }
            columnNumber /= 26;
        }
        return res.reverse().toString();
    }

    // 67. 二进制求和
    public String addBinary(String a, String b) {
        StringBuilder res = new StringBuilder();
        int i = a.length() - 1;
        int j = b.length() - 1;
        int tmp = 0;
        while (i >= 0 || j >= 0) {
            if (i >= 0) {
                tmp += a.charAt(i) - '0';
                i--;
            }
            if (j >= 0) {
                tmp += b.charAt(j) - '0';
                j--;
            }
            res.append(tmp % 2);
            tmp /= 2;
        }
        if (tmp == 1) {
            res.append(1);
        }

        return res.reverse().toString();
    }

    // 238. 除自身以外数组的乘积
    public int[] productExceptSelf(int[] nums) {
        int[] res = new int[nums.length];
        Arrays.fill(res, 1);
        for (int i = 1; i < nums.length; i++) {
            res[i] = res[i - 1] * nums[i - 1];
        }
        int tmp = nums[nums.length - 1];
        for (int i = nums.length - 2; i >= 0; i--) {
            res[i] *= tmp;
            tmp *= nums[i];
        }
        return res;
    }

    // 优化
    public int[] productExceptSelf2(int[] nums) {
        int len = nums.length;
        int[] res = new int[len];
        Arrays.fill(res, 1);
        int left = 1;
        int right = 1;
        for (int i = 0; i < len; i++) {
            res[i] *= left;
            left *= nums[i];

            res[len - i - 1] *= right;
            right *= nums[len - i - 1];
        }
        return res;
    }

    // 462. 最少移动次数使数组元素相等 II
    // 这题看一下为啥最优解是中位数的leetcode收藏，思路非常棒，而且不涉及数学，哭唧唧
    // 这题再另外思考一下，快排寻找第K大的数
    public int minMoves2(int[] nums) {
        int sum = 0;
        Arrays.sort(nums);
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            sum += (nums[right] - nums[left]);
            left++;
            right--;
        }
        return sum;
    }

    // 169. 多数元素
    // 多数投票算法
    // 这个算法只有在一半以上才有用
    public int majorityElement(int[] nums) {
        int res = 0;
        int count = 0;
        for (int num : nums) {
            if (count == 0) {
                res = num;
            }
            if (res == num) {
                count++;
            } else {
                count--;
            }
        }
        return res;
    }

    // 462. 最少移动次数使数组元素相等 II
    // (rand_X() - 1) × Y + rand_Y() ==> 可以等概率的生成[1, X * Y]范围的随机数
    // rand_nX() % X + 1可以生成[1, X]范围的随机数
    public int rand10() {
        int res = (rand7() - 1) * 7 + rand7();
        while (res > 40) {
            res = (rand7() - 1) * 7 + rand7();
        }
        return res % 10 + 1;
    }

    // 公式优化算法
    public int rand103() {
        do {
            int res = (rand7() - 1) * 7 + rand7();
            if (res <= 40) {
                return res % 10 + 1;
            }
            res -= 40; //rand9
            res = (res - 1) * 7 + rand7();
            if (res <= 60) {
                return res % 10 + 1;
            }
            res -= 60; //rand3
            res = (res - 1) * 7 + rand7();
            if (res <= 20) {
                return res % 10 + 1;
            }
        } while (true);
    }

    // 1/10 = 1/2 * 1/5
    public int rand102() {
        int a = rand7();
        int b = rand7();
        while (a > 6) {
            a = rand7();
        }
        while (b > 5) {
            b = rand7();
        }
        if (a % 2 == 1) {
            return 5 + b;
        }
        return b;
    }

    int rand7() {
        return random.nextInt(7) + 1;
    }

    // 202. 快乐数
    // 我是傻逼
    // 这里有一个数学知识点，一个数经过各位平方和不断计算下去
    // 必然会进入循环，具体的证明涉及到数学问题，我tm看不懂
    public boolean isHappy(int n) {
        int slow = n;
        int fast = n;
        do {
            slow = bitSquareSum(slow);
            fast = bitSquareSum(fast);
            fast = bitSquareSum(fast);
        } while (slow != fast);
        return slow == 1;
    }

    int bitSquareSum(int n) {
        int res = 0;
        while (n > 0) {
            int tmp = n % 10;
            res = res + tmp * tmp;
            n /= 10;
        }
        return res;
    }

    public static void main(String[] args) {
        MathProblem mathProblem = new MathProblem();
        System.out.println(mathProblem.addBinary("1010", "1011"));
    }
}

// 384. 打乱数组
class Solution {
    int[] nums;
    int[] origin;
    Random random = new Random();

    public Solution(int[] nums) {
        this.nums = nums;
        this.origin = nums.clone();
    }

    /**
     * Resets the array to its original configuration and return it.
     */
    public int[] reset() {
        nums = origin.clone();
        return origin;
    }

    /**
     * Returns a random shuffling of the array.
     */
    public int[] shuffle() {
        for (int i = 0; i < nums.length; i++) {
            swap(i, rand(i));
        }
        return nums;
    }

    public int rand(int min) {
        // random.nextInt:[0,n)
        int max = nums.length;
        return random.nextInt(max - min) + min;
    }

    public void swap(int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
}

// 528. 按权重随机选择
class Solution2 {
    int[] w;
    int[] prefixSum;
    Random random = new Random();

    public Solution2(int[] w) {
        this.w = w;
        this.prefixSum = w.clone();
        for (int i = 1; i < prefixSum.length; i++) {
            prefixSum[i] += prefixSum[i - 1];
        }
    }

    public int pickIndex() {
        int rand = random.nextInt(prefixSum[prefixSum.length - 1]) + 1;
        return binarySearch(rand);
    }

    public int binarySearch(int target) {
        int left = 0, right = w.length - 1;
        while (left < right) {
            int mid = (left + right) / 2;
            if (target < prefixSum[mid]) {
                right = mid;
            } else if (target > prefixSum[mid]) {
                left = mid + 1;
            } else {
                return mid;
            }
        }
        return right;
    }
}

// 382. 链表随机节点
// 这道题，emmmm，总之记住水库算法吧
class Solution3 {
    ListNode head;
    Random random = new Random();

    /**
     * @param head The linked list's head.
     *             Note that the head is guaranteed to be not null, so it contains at least one node.
     */
    public Solution3(ListNode head) {
        this.head = head;
    }

    /**
     * Returns a random node's value.
     */
    public int getRandom() {
        ListNode node = head;
        int res = node.val;
        int step = 1;
        while (node != null) {
            if (random.nextInt(step) + 1 == step) {
                res = node.val;
            }
            step++;
            node = node.next;
        }
        return res;
    }
}

