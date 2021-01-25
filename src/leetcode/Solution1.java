package leetcode;

import java.util.*;

public class Solution1 {
    // 4. 寻找两个正序数组的中位数
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int len1 = nums1.length;
        int len2 = nums2.length;
        int len = len1 + len2;
        double median;
        if (len % 2 == 1) {
            // 总长度为奇数
            int index = len / 2 + 1;
            median = getKthElement(nums1, nums2, index);
        } else {
            // 总长度为偶数
            int index1 = len / 2, index2 = len / 2 + 1;
            median = (getKthElement(nums1, nums2, index1) + getKthElement(nums1, nums2, index2)) / 2.0;
        }
        return median;
    }

    public int getKthElement(int[] nums1, int[] nums2, int k) {
        int len1 = nums1.length;
        int len2 = nums2.length;
        int index1 = 0, index2 = 0;

        while (true) {
            if (index1 == len1) {
                return nums2[index2 + k - 1];
            }

            if (index2 == len2) {
                return nums1[index1 + k - 1];
            }

            if (k == 1) {
                return Math.min(nums1[index1], nums2[index2]);
            }

            int half = k / 2;
            int newIndex1 = Math.min(index1 + half, len1) - 1;
            int newIndex2 = Math.min(index2 + half, len2) - 1;
            if (nums1[newIndex1] <= nums2[newIndex2]) {
                k = k - (newIndex1 - index1 + 1);
                index1 = newIndex1 + 1;
            } else {
                k = k - (newIndex2 - index2 + 1);
                index2 = newIndex2 + 1;
            }
        }
    }

    // 53. 最大子序和
    // 贪心
    public int maxSubArray1(int[] nums) {
        int len = nums.length;
        int max = nums[0];
        int pre = nums[0];
        for (int i = 1; i < len; i++) {
            int cur = nums[i];
            if (pre <= 0) {
                pre = cur;
            } else {
                pre += cur;
            }
            max = Math.max(pre, max);
        }
        return max;
    }

    // 88. 合并两个有序数组
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int index1 = m - 1;
        int index2 = n - 1;
        int p = nums1.length - 1;
        while (true) {
            if (index1 < 0 || index2 < 0) {
                break;
            }
            if (nums2[index2] > nums1[index1]) {
                nums1[p] = nums2[index2];
                index2--;
            } else {
                nums1[p] = nums1[index1];
                index1--;
            }
            p--;
        }
    }

    // 1046. 最后一块石头的重量
    public int lastStoneWeight(int[] stones) {
        // 通过完全二叉树实现的小顶堆
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>((a, b) -> b - a);
        for (int stone : stones) {
            priorityQueue.add(stone);
        }
        while (priorityQueue.size() > 1) {
            int val1 = priorityQueue.poll();
            int val2 = priorityQueue.poll();
            int dis = val1 - val2;
            if (dis > 0) {
                priorityQueue.add(dis);
            }
        }
        return priorityQueue.isEmpty() ? 0 : priorityQueue.poll();
    }

    // 509. 斐波那契数
    public int fib(int n) {
        if (n == 0) {
            return 0;
        } else if (n == 1) {
            return 1;
        } else {
            return fib(n - 1) + fib(n - 2);
        }
    }

    // 547. 省份数量
    // 广度优先搜素
    public int findCircleNum(int[][] isConnected) {
        int n = isConnected.length;
        int res = 0;
        boolean[] visited = new boolean[n];
        for (int row = 0; row < n; row++) {
            for (int col = row + 1; col < n; col++) {
                if(isConnected[row][col] == 1 && !visited[col]){
                    res ++;
                    visited[col] = true;
                }
            }
        }
        System.out.println("res=" + res);
        return n - res;
    }

    public static void main(String[] args) {
        Solution1 s1 = new Solution1();

        /*
        int[] nums1 = {1,2};
        int[] nums2 = {3,4};
        double res = s1.findMedianSortedArrays(nums1, nums2);
        System.out.println(res);

        int[] nums = {-2,1,-3,4,-1,2,1,-5,4};
        System.out.println(s1.maxSubArray1(nums));

        int[] nums1 = {0};
        int[] nums2 = {1};
        int m = 0, n = 1;
        s1.merge(nums1, m, nums2, n);
        System.out.println(Arrays.toString(nums1));

        int[] stones = {2, 7, 4, 1, 8, 1};
        int res = s1.lastStoneWeight(stones);
        System.out.println(res);

        System.out.println(s1.fib(5));
        */
        int[][] isConnected = {{1,1,0,0,0,0,0,1,0,0,0,0,0,0,0},
                               {1,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
                               {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0},
                               {0,0,0,1,0,1,1,0,0,0,0,0,0,0,0},
                               {0,0,0,0,1,0,0,0,0,1,1,0,0,0,0},
                               {0,0,0,1,0,1,0,0,0,0,1,0,0,0,0},
                               {0,0,0,1,0,0,1,0,1,0,0,0,0,1,0},
                               {1,0,0,0,0,0,0,1,1,0,0,0,0,0,0},
                               {0,0,0,0,0,0,1,1,1,0,0,0,0,1,0},
                               {0,0,0,0,1,0,0,0,0,1,0,1,0,0,1},
                               {0,0,0,0,1,1,0,0,0,0,1,1,0,0,0},
                               {0,0,0,0,0,0,0,0,0,1,1,1,0,0,0},
                               {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0},
                               {0,0,0,0,0,0,1,0,1,0,0,0,0,1,0},
                               {0,0,0,0,0,0,0,0,0,1,0,0,0,0,1}};
        int res = s1.findCircleNum(isConnected);
        System.out.println(res);
    }
}
