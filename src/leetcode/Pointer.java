package leetcode;

import java.util.*;

class ListNode {
    int val;
    ListNode next;

    ListNode() {
    }

    ListNode(int x) {
        val = x;
        next = null;
    }

    ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }
}

public class Pointer {
    // 双指针
    // 167. 两数之和 II - 输入有序数组
    public int[] twoSum(int[] numbers, int target) {
        int len = numbers.length;
        int i = 0;
        int j = len - 1;
        int[] res = new int[2];
        while (i < j) {
            int sum = numbers[i] + numbers[j];
            if (sum == target) {
                res[0] = i + 1;
                res[1] = j + 1;
                return res;
            } else if (sum < target) {
                i++;
            } else {
                j--;
            }
        }
        return res;
    }

    // 88. 合并两个有序数组
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int index1 = m - 1;
        int index2 = n - 1;
        int p = nums1.length - 1;

        while (index1 >= 0 && index2 >= 0) {
            if (nums1[index1] >= nums2[index2]) {
                nums1[p] = nums1[index1];
                index1--;
            } else {
                nums1[p] = nums2[index2];
                index2--;
            }
            p--;
        }

        while (index2 >= 0) {
            nums1[p--] = nums2[index2--];
        }
    }

    // 142. 环形链表 II
    public ListNode detectCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while (true) {
            if (fast == null || fast.next == null) {
                return null;
            }
            slow = slow.next;
            fast = fast.next.next;
            if (fast == slow) {
                break;
            }
        }
        fast = head;
        while (fast != slow) {
            fast = fast.next;
            slow = slow.next;
        }
        return fast;
    }

    // 76. 最小覆盖子串
    public String minWindow(String s, String t) {
        // s = "ADOBECODEBANC", t = "ABC"
        int lenS = s.length();
        int lenT = t.length();
        if (lenS == 0 || lenS < lenT || lenT == 0) {
            return "";
        }
        // 目标字符串预处理
        Map<Character, Integer> target = new HashMap<>();
        for (char c : t.toCharArray()) {
            target.put(c, target.getOrDefault(c, 0) + 1);
        }
        Map<Character, Integer> window = new HashMap<>();
        int left = 0;
        int right = 0;
        int count = 0;
        int step = lenS + 1;
        int start = 0;
        char[] charS = s.toCharArray();
        while (right < lenS) {
            char rightC = charS[right];
            right++;
            if (target.containsKey(rightC)) {
                window.put(rightC, window.getOrDefault(rightC, 0) + 1);
                if (target.get(rightC).equals(window.get(rightC))) {
                    count++;
                }
            }
            while (count == target.size()) {
                if (right - left < step) {
                    start = left;
                    step = right - left;
                }
                char leftC = charS[left];
                left++;
                if (target.containsKey(leftC)) {
                    if (window.get(leftC).equals(target.get(leftC))) {
                        count--;
                    }
                    window.put(leftC, window.get(leftC) - 1);
                }
            }
        }

        return start + step > lenS ? "" : s.substring(start, start + step);
    }

    // 633. 平方数之和
    public boolean judgeSquareSum(int c) {
        if (c < 0) {
            return false;
        }
        int i = 0;
        int j = (int) Math.sqrt(c);
        while (i <= j) {
            int num = i * i + j * j;
            if (num == c) {
                return true;
            } else if (num < c) {
                i++;
            } else {
                j--;
            }
        }
        return false;
    }

    // 680. 验证回文字符串 Ⅱ
    public boolean validPalindrome(String s) {
        int len = s.length();
        if (len == 0 || len == 1) {
            return true;
        }
        int left = 0;
        int right = len - 1;
        while (left < right) {
            if (s.charAt(left) == s.charAt(right)) {
                left++;
                right--;
            } else {
                return isValid(s, left + 1, right) || isValid(s, left, right - 1);
            }
        }

        return true;
    }

    public boolean isValid(String s, int left, int right) {
        while (left < right) {
            if (s.charAt(left) != s.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }

    // 524. 通过删除字母匹配到字典里最长单词
    public String findLongestWord(String s, List<String> d) {
        // s = "abpcplea", d = ["ale","apple","monkey","plea"]
        int len = s.length();
        if (len == 0) {
            return "";
        }
        List<String> res = new ArrayList<>();
        for (String son : d) {
            int j = 0;
            for (int i = 0; i < len; i++) {
                if (s.charAt(i) == son.charAt(j)) {
                    j++;
                }
                if (j == son.length()) {
                    res.add(son);
                    break;
                }
            }
        }
        res.sort(((o1, o2) -> o1.length() == o2.length() ? o1.compareTo(o2) : o2.length() - o1.length()));

        return res.size() == 0 ? "" : res.get(0);
    }

    // 340. 需要会员解锁，以后有机会再写吧。

    public static void main(String[] args) {
        Pointer pointer = new Pointer();

        int[] numbers = {2, 7, 11, 15};
        int target = 9;
        // System.out.println(Arrays.toString(pointer.twoSum(numbers, target)));

        int[] nums1 = {1, 2, 3, 0, 0, 0};
        int m = 3;
        int[] nums2 = {2, 5, 6};
        int n = 3;
        // pointer.merge(nums1, m, nums2, n);
        // System.out.println(Arrays.toString(nums1));

        // String s = "a";
        String t = "b";
        // System.out.println(pointer.minWindow(s, t)); //BANC

        int c = 1000;
        // System.out.println(pointer.judgeSquareSum(c));

        // String s = "aguokepatgbnvfqmgmlcupuufxoohdfpgjdmysgvhmvffcnqxjjxqncffvmhvgsymdjgpfdhooxfuupuculmgmqfvnbgtapekouga";
        // System.out.println(pointer.validPalindrome(s));

        String s = "bab";
        List<String> d = new ArrayList<>();
        d.add("ba");
        d.add("ab");
        d.add("a");
        d.add("b");
        System.out.println(pointer.findLongestWord(s, d));
    }
}
