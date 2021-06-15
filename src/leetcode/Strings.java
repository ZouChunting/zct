package leetcode;

import java.util.*;

public class Strings {
    // 字符串比较
    // 242. 有效的字母异位词
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }
        char[] cs = s.toCharArray();
        Arrays.sort(cs);
        char[] ct = t.toCharArray();
        Arrays.sort(ct);
        return Arrays.equals(cs, ct);
    }

    // 205. 同构字符串
    public boolean isIsomorphic(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }
        for (int i = 0; i < s.length(); i++) {
            if (s.indexOf(s.charAt(i)) != t.indexOf(t.charAt(i))) {
                return false;
            }
        }
        return true;
    }

    // 647. 回文子串
    public int countSubstrings(String s) {
        char[] cs = s.toCharArray();
        int res = 0;
        for (int i = 0; i < cs.length; i++) {
            // 以每一点为中心轴扩散
            res += spreadString(cs, i, i); // 奇数回文串
            res += spreadString(cs, i, i + 1); // 偶数回文串
        }
        return res;
    }

    int spreadString(char[] cs, int left, int right) {
        int count = 0;
        while (left >= 0 && right < cs.length && cs[left] == cs[right]) {
            left--;
            right++;
            count++;
        }
        return count;
    }

    // 696. 计数二进制子串
    // 立足于当前字符，如果之前的相异字符的连续个数比当前字符的连续个数大于或相等
    // 说明可以截取成相等的字符串
    public int countBinarySubstrings(String s) {
        char[] cs = s.toCharArray();
        int res = 0;
        int pre = 0;
        int cur = 1;

        for (int i = 1; i < s.length(); i++) {
            if (cs[i] == cs[i - 1]) {
                cur++;
            } else {
                pre = cur;
                cur = 1;
            }
            if (pre >= cur) {
                res++;
            }
        }
        return res;
    }

    // 字符串理解
    // 227. 基本计算器 II
    public int calculate(String s) {
        s = s.replace(" ", "");
        String[] plusList = s.split("\\+");
        int plusCount = 0;
        for (String plus : plusList) {
            String[] subList = plus.split("-");
            int subCount = calME(subList[0]);
            for (int i = 1; i < subList.length; i++) {
                // 现在只剩乘除了，先把乘除计算出来
                int meCount = calME(subList[i]);
                subCount -= meCount;
            }
            plusCount += subCount;
        }

        return plusCount;
    }

    int calME(String s) {
        int res = 1;
        char sym = '*';
        int index = 0;
        int cur = 0;
        while (index < s.length()) {
            // 是数字
            if (Character.isDigit(s.charAt(index))) {
                cur = cur * 10 + (s.charAt(index) - '0');
            } else {
                res = sym == '*' ? res * cur : res / cur;
                sym = s.charAt(index);
                cur = 0;
            }
            index++;
        }
        res = sym == '*' ? res * cur : res / cur;

        return res;
    }

    // 字符串匹配
    // 28. 实现 strStr()
    // md，kmp怎么那么难
    public int strStr(String haystack, String needle) {
        if (needle.length() == 0) {
            return 0;
        }
        if (haystack.length() == 0 || haystack.length() < needle.length()) {
            return -1;
        }
        // 计算前缀数组
        int[] dp = new int[needle.length()];
        nextArray(needle.toCharArray(), dp);
        // 匹配

        return -1;
    }

    void nextArray(char[] cs, int[] dp) {
        dp[0] = 0;
        for (int i = 1, prefixSum = 0; i < cs.length; i++) {
            // 举例：ABABC，当运算到C时，先比较C(下标:4)和A(下标:2)是否相等
            // 显然不等，那么再将C和
            // 举例：AAAAC
            while (prefixSum > 0 && cs[i] != cs[prefixSum]) {
                prefixSum = dp[prefixSum] - 1;
            }
            if (cs[i] == cs[prefixSum]) {
                prefixSum++;
            }
            dp[i] = prefixSum;
        }
    }

    // 409. 最长回文串
    // "abccccdd" "dccaccd"
    public int longestPalindrome(String s) {
        char[] cs = s.toCharArray();
        Set<Character> set = new HashSet<>();
        int res = 0;
        for (char c : cs) {
            if (set.contains(c)) {
                res += 2;
                set.remove(c);
            } else {
                set.add(c);
            }
        }
        res = res + (set.isEmpty() ? 0 : 1);
        return res;
    }

    // 3. 无重复字符的最长子串
    public int lengthOfLongestSubstring(String s) {
        if (s.length() < 2) {
            return s.length();
        }
        int res = 1;
        char[] cs = s.toCharArray();
        int left = 0;
        for (int i = 0; i < s.length(); i++) {
            for (int j = left; j < i; j++) {
                if (cs[i] == cs[j]) {
                    res = Math.max(res, i - left);
                    left = j + 1;
                    break;
                }
            }
        }

        return Math.max(res, s.length() - left);
    }

    // 5. 最长回文子串
    public String longestPalindrome2(String s) {
        char[] cs = s.toCharArray();

        return "";
    }

    public static void main(String[] args) {
        Strings strings = new Strings();

        String s = "3/2";
        // String t = "bar";
        // System.out.println(strings.isIsomorphic(s, t));

        // System.out.println(strings.countSubstrings(s));

        // System.out.println(strings.calME(s));

        // System.out.println(strings.longestPalindrome("abccccdd"));

        System.out.println(strings.lengthOfLongestSubstring("abc"));
    }
}
