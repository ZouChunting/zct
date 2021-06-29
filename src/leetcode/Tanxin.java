package leetcode;

import java.util.*;

public class Tanxin {
    // 贪心算法
    // 455. 分发饼干
    public int findContentChildren(int[] g, int[] s) {
        // 数组排序
        Arrays.sort(g);  // 孩子
        Arrays.sort(s);  // 饼干
        int lenG = g.length;
        int lenS = s.length;
        int child = 0;
        int cookie = 0;
        while (child < lenG && cookie < lenS) {
            if (g[child] <= s[cookie]) {
                child++;
            }
            cookie++;
        }

        return child;
    }

    // 135. 分发糖果
    public int candy(int[] ratings) {
        int len = ratings.length;
        int[] cans = new int[len];
        Arrays.fill(cans, 1);
        // 左->右 保证相邻右比左大 增长趋势
        for (int i = 1; i < len; i++) {
            if (ratings[i] > ratings[i - 1]) {
                cans[i] = cans[i - 1] + 1;
            }
        }
        // 右->左 保证相邻左比右大
        for (int i = len - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1] && cans[i] <= cans[i + 1]) {
                cans[i] = cans[i + 1] + 1;
            }
        }

        return Arrays.stream(cans).sum();
    }

    // 435. 无重叠区间
    public int eraseOverlapIntervals(int[][] intervals) {
        int res = 0;
        int len = intervals.length;
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] == o2[0] ? o1[0] - o2[0] : o1[1] - o2[1];
            }
        });
        int i = 0;
        int j = 1;
        while (j < len) {
            if (intervals[i][1] > intervals[j][0]) {
                res++;
            } else {
                i = j;
            }
            j++;
        }

        return res;
    }

    // 605. 种花问题
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        int len = flowerbed.length;
        for (int i = 0; i < len && n > 0; ) {
            if (flowerbed[i] == 1) {
                i = i + 2;
            } else if (i == len - 1 || flowerbed[i + 1] == 0) {
                n--;
                i = i + 2;
            } else {
                i = i + 3;
            }
        }

        return n <= 0;
    }

    // 452. 用最少数量的箭引爆气球
    public int findMinArrowShots(int[][] points) {
        int len = points.length;
        if (len == 0) {
            return 0;
        }
        // 数组排序
        Arrays.sort(points, ((o1, o2) -> o1[1] < o2[1] ? -1 : 1));
        int res = 1;
        int pre = points[0][1];
        for (int i = 1; i < len; i++) {
            if (points[i][0] > pre) {
                res++;
                pre = points[i][1];
            }
        }

        return res;
    }

    // 763. 划分字母区间
    public List<Integer> partitionLabels(String S) {
        // "ababcbaca", "defegde", "hijhklij"
        int len = S.length();
        List<Integer> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }
        if (len == 1) {
            res.add(1);
            return res;
        }
        // 记录字符最后一次出现的位置
        Map<Character, Integer> last = new HashMap<>();
        char[] chars = S.toCharArray();
        for (int i = 0; i < len; i++) {
            last.put(chars[i], i);
        }
        int num = 0;
        int index = last.get(chars[0]);
        for (int i = 0; i < len; i++) {
            num++;
            index = Math.max(index, last.get(chars[i]));
            if (i == index) {
                res.add(num);
                num = 0;
            }
        }

        return res;
    }

    // 122. 买卖股票的最佳时机 II
    public int maxProfit(int[] prices) {
        // 7,1,5,3,6,4
        int res = 0;
        int len = prices.length;
        for (int i = 1; i < len; i++) {
            int diff = prices[i] - prices[i - 1];
            if (diff > 0) {
                res += diff;
            }
        }

        return res;
    }

    // 406. 根据身高重建队列
    public int[][] reconstructQueue(int[][] people) {
        int len = people.length;
        if (len == 0 || len == 1) {
            return people;
        }
        Arrays.sort(people, (o1, o2) -> {
            if (o1[0] != o2[0]) {
                return o2[0] - o1[0];
            } else {
                return o1[1] - o2[1];
            }
        });
        List<int[]> list = new ArrayList<>();
        for (int i = 0; i < len; i++) {
            int[] p = people[i];
            if (p[1] >= i) {
                list.add(p);
            } else {
                list.add(p[1], p);
            }
        }

        return list.toArray(new int[len][2]);
    }

    // 665. 非递减数列
    // 前 <= 后
    public boolean checkPossibility(int[] nums) {
        int len = nums.length;
        if (len == 0 || len == 1) {
            return true;
        }
        int change = 0;
        for (int i = 1; i < len; i++) {
            if (nums[i] < nums[i - 1]) {
                change++;
                if (change > 1) {
                    return false;
                }
                // 贪心策略i-2,i-1,i（关键：i-2是否存在）
                // 1. i-2不存在，i-1置为i
                // 2. i-2存在，且小于等于i，i-1置为i-2
                // 3. i-2存在，且大于i，i置为i-1
                if (i - 2 < 0) {
                    nums[i - 1] = nums[i];
                } else {
                    if (nums[i - 2] <= nums[i]) {
                        nums[i - 1] = nums[i - 2];
                    } else {
                        nums[i] = nums[i - 1];
                    }
                }
            }
        }
        return true;
    }

    public static void main(String[] args) {
        Tanxin tanxin = new Tanxin();

        int[] g = {1, 2, 3};
        int[] s = {1, 1};
        // tanxin.findContentChildren(g,s);

        int[] ratings = {1, 0, 2};
        // System.out.println(tanxin.candy(ratings));

        int[][] intervals = {{1, 2}, {2, 3}};
        // System.out.println(tanxin.eraseOverlapIntervals(intervals));

        int[] flowerbed = {1, 0, 0, 0, 1};
        // System.out.println(tanxin.canPlaceFlowers(flowerbed, 2));

        int[][] points = {{10, 16}, {2, 8}, {1, 6}, {7, 12}};
        // System.out.println(tanxin.findMinArrowShots(points));

        String S = "ababcbacadefegdehijhklij";
        // System.out.println(tanxin.partitionLabels(S));

        int[][] people = {{7, 0}, {4, 4}, {7, 1}, {5, 0}, {6, 1}, {5, 2}};
        // System.out.println(Arrays.deepToString(tanxin.reconstructQueue(people)));

        int[] nums = {2, 3, 3, 2, 4};
        System.out.println(tanxin.checkPossibility(nums));
    }
}
