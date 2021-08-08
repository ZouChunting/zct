package leetcode;

import java.util.Arrays;

public class BitOperation {
    // 位运算

    // &与：都是1，结果才是1，否则为0
    // |或：有一个数字的值是1，则是1，否则是0
    // ^异或：相同的得0，不同的得1
    // ~非：1变0，0变1
    // <<左移：在当前类型的数值范围内，每往左移动1位，得到的数值翻倍，移动n位，得到为a*2**n
    // >>右移：a右移n位，等于（int）(a/2**n)

    // 二进制取反
    // 举例：5转换为二进制数为： 0000 0000 0000 0101得到二进制数
    // 每一位取反： 1111 1111 1111 1010得到最终结果的补码
    // 取补码： 1000 0000 0000 0110得到最终结果的原码
    // 转换为十进制数：-6
    // 则 5 取反为 -6

    // 461. 汉明距离
    public int hammingDistance(int x, int y) {
        int diff = x ^ y, res = 0;
        while (diff != 0) {
            res += diff & 1;  // 判断末位是不是1
            diff >>= 1; // 右移，即去掉已经计算过的末位
        }

        return res;
    }

    // 190. 颠倒二进制位
    public int reverseBits(int n) {
        int res = 0;
        for (int i = 0; i < 32; i++) {
            res <<= 1;  // 左移，空出末位
            res += n & 1;  // 取原数末位，放在结果的开头
            n >>= 1; // 右移，去除已计算的末位
        }
        return res;
    }

    // 136. 只出现一次的数字
    public int singleNumber(int[] nums) {
        int res = 0;
        for (int num : nums) {
            res ^= num;
        }

        return res;
    }

    // 342. 4的幂
    // 1431655765  10101...101
    public boolean isPowerOfFour(int n) {
        if (n < 0) {
            return false;
        }
        int flag1 = n & (n - 1); // 保证n只有一个1，如果n有多于1个1，那么必定结果不为0；
        int flag2 = n & 1431655765;  // 保证n在奇数为有1
        return flag1 == 0 && flag2 != 0;
    }

    // 318. 最大单词长度乘积
    public int maxProduct(String[] words) {
        int len = words.length;
        int[] hash = new int[len];
        for (int i = 0; i < len; i++) {
            char[] word = words[i].toCharArray();
            int tmp = 0;
            for (char c : word) {
                tmp |= 1 << c - 'a';
            }
            hash[i] = tmp;
        }
        int res = 0;
        for (int i = 0; i < len - 1; i++) {
            for (int j = i + 1; j < len; j++) {
                if ((hash[i] & hash[j]) == 0) {
                    res = Math.max(res, words[i].length() * words[j].length());
                }
            }
        }
        return res;
    }

    // 338. 比特位计数
    public int[] countBits(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 0;
        for (int i = 1; i <= n; i++) {
            // 如果是偶数，那么含有1的个数，跟右移数的一样
            // 如果是奇数，前一位偶数的值加1
            dp[i] = (i & 1) == 0 ? dp[i >> 1] : dp[i - 1] + 1;
        }
        return dp;
    }

    // 268. 丢失的数字
    // 高斯求和公式
    public int missingNumber(int[] nums) {
        int len = nums.length;
        if (len == 0) {
            return 0;
        }
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        return len * (len + 1) / 2 - sum;
    }

    // 位运算
    public int missingNumber2(int[] nums) {
        int len = nums.length;
        if (len == 0) {
            return 0;
        }
        int sum = 0;
        for (int i = 0; i < len; i++) {
            // 相同的数字异或为0
            sum = sum ^ i ^ nums[i];
        }
        return sum ^ len;
    }

    // 693. 交替位二进制数
    public boolean hasAlternatingBits(int n) {
        n = n ^ (n >> 1);
        return (n & (n + 1)) == 0;
    }

    // 476. 数字的补数
    public int findComplement(int num) {
        int res = 0;
        int count = 0;
        while (num != 0) {
            // 查看末位是否为0，若为0则补1
            if ((num & 1) == 0) {
                res += (1 << count);
            }
            count++;
            num = num >> 1;
        }

        return res;
    }

    // 260. 只出现一次的数字 III
    // 我想你需要去了解一下，取反是什么操作以及各种二进制的计算
    public int[] singleNumber2(int[] nums) {
        int tmp = 0;
        for (int num : nums) {
            tmp ^= num;
        }
        // 取末位1
        int diff = tmp & (-tmp);
        int[] res = new int[2];
        for (int num : nums) {
            if ((num & diff) == 0) {
                res[0] ^= num;
            } else {
                res[1] ^= num;
            }
        }
        return res;
    }

    public static void main(String[] args) {
        BitOperation bitOperation = new BitOperation();
        System.out.println(bitOperation.hammingDistance(1, 4));
    }
}
