package leetcode;

import com.sun.tools.javac.code.Attribute;
import javafx.util.Pair;

import javax.swing.plaf.IconUIResource;
import javax.tools.Diagnostic;
import java.util.*;
import java.util.function.Function;

public class Structure {
    // 数组
    // 448. 找到所有数组中消失的数字
    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> res = new ArrayList<>();
        for (int num : nums) {
            int index = Math.abs(num) - 1;
            nums[index] = nums[index] > 0 ? -nums[index] : nums[index];
        }
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) {
                res.add(i + 1);
            }
        }
        return res;
    }

    // 48. 旋转图像
    // 能旋转90度说明长宽一致
    public void rotate(int[][] matrix) {
        int len = matrix.length;
        int tmp;
        for (int i = 0; i < len / 2; i++) {
            for (int j = i; j < len - i - 1; j++) {
                tmp = matrix[i][j];
                matrix[i][j] = matrix[len - j - 1][i];
                matrix[len - j - 1][i] = matrix[len - i - 1][len - j - 1];
                matrix[len - i - 1][len - j - 1] = matrix[j][len - i - 1];
                matrix[j][len - i - 1] = tmp;
            }
        }
    }

    // 240. 搜索二维矩阵 II
    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length;
        int n = matrix[0].length;
        int i = 0, j = n - 1;
        while (i < m && j >= 0) {
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] < target) {
                i++;
            } else {
                j--;
            }
        }

        return false;
    }

    // 769. 最多能完成排序的块
    public int maxChunksToSorted(int[] arr) {
        int len = arr.length;
        int res = 0;
        int max = 0;
        for (int i = 0; i < len; i++) {
            // 计算当前最大值
            max = Math.max(max, arr[i]);
            if (max == i) {
                res++;
            }
        }
        return res;
    }

    // 栈和队列
    // 20. 有效的括号
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char symbol = s.charAt(i);
            if (symbol == '(' || symbol == '{' || symbol == '[') {
                stack.push(symbol);
            } else {
                if (symbol == ')') {
                    if (stack.empty() || stack.pop() != '(') {
                        return false;
                    }
                } else if (symbol == '}') {
                    if (stack.empty() || stack.pop() != '{') {
                        return false;
                    }
                } else {
                    if (stack.empty() || stack.pop() != '[') {
                        return false;
                    }
                }

            }
        }
        return stack.empty();
    }

    // 单调栈
    // 739. 每日温度
    public int[] dailyTemperatures(int[] T) {
        int[] res = new int[T.length];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < T.length; i++) {
            while (!stack.empty()) {
                int t = stack.peek();
                if (T[i] <= T[t]) {
                    break;
                }
                res[stack.peek()] = i - stack.peek();
                stack.pop();
            }
            stack.push(i);
        }

        return res;
    }

    // 优先队列
    // 23. 合并K个升序链表
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists.length == 0) {
            return null;
        }
        PriorityQueue<ListNode> pq = new PriorityQueue<>((o1, o2) -> o1.val - o2.val);
        for (ListNode node : lists) {
            if (node != null) {
                pq.add(node);
            }
        }
        ListNode res = new ListNode(0);
        res.next = pq.poll();
        ListNode next = res.next;
        if (next != null && next.next != null) {
            pq.offer(next.next);
        }
        while (!pq.isEmpty()) {
            // 剩余最小
            ListNode tmp = pq.poll();
            next.next = tmp;
            if (tmp.next != null) {
                pq.offer(tmp.next);
            }
            next = next.next;
        }
        return res.next;
    }

    // 218. 天际线问题
    // 不会写，tmd连答案都看不懂，先空着吧，嗐
    public List<List<Integer>> getSkyline(int[][] buildings) {
        List<List<Integer>> res = new ArrayList<>();
        return res;
    }

    // 双端队列
    // 239. 滑动窗口最大值
    public int[] maxSlidingWindow(int[] nums, int k) {
        int len = nums.length - k + 1;
        int[] res = new int[len];
        // 存的不是数，而是下标
        Deque<Integer> dq = new ArrayDeque<>();
        for (int i = 0; i < nums.length; i++) {
            // i - k即为控制双端队列的窗口，
            // 首先，队列中的坐标是按先后顺序排的
            // 当坐标超过窗口，就把最早的移除出去
            if (!dq.isEmpty() && dq.peekFirst() == i - k) {
                dq.pollFirst();
            }
            // nums[i]是新数字，从后往前遍历已有队列，
            // 已知队列中坐标必然小于新坐标，那么当队列中的值也小于新值时，
            // 就没有存在必要，直接移除
            while (!dq.isEmpty() && nums[dq.peekLast()] <= nums[i]) {
                dq.pollLast();
            }
            // 经过上述处理，队列要么为空，要么全是大于当前数的。
            // 因此可以保证，大的数永远在最前面
            dq.offerLast(i);
            if (i >= k - 1) {
                res[i - k + 1] = nums[dq.peekFirst()];
            }
        }
        return res;
    }

    // 1. 两数之和
    public int[] twoSum(int[] nums, int target) {
        int[] res = new int[2];
        // 值：坐标
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                res[0] = map.get(target - nums[i]);
                res[1] = i;
                return res;
            }
            map.put(nums[i], i);
        }
        return res;
    }

    // 128. 最长连续序列
    public int longestConsecutive(int[] nums) {
        int res = 0;
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        for (int num : nums) {
            if (set.remove(num)) {
                int pre = num, next = num;
                int len = 1;
                while (set.remove(pre - 1)) {
                    pre--;
                }
                len = len + num - pre;
                while (set.remove(next + 1)) {
                    next++;
                }
                len = len + next - num;
                res = Math.max(res, len);
            }
        }

        return res;
    }

    // 149. 直线上最多的点数
    // 这道题目貌似没有看起来那么难
    public int maxPoints(int[][] points) {
        // 斜率：个数
        Map<Double, Integer> map = new HashMap<>();
        int max = 0, same, same_y;
        for (int i = 0; i < points.length; i++) {
            same = 1;
            same_y = 1;
            // 分三种斜率讨论
            // 1.在同一横轴线上（认为此种不可计算）
            // 2.重合（重合的情况是需要叠加到其他两种情况上的）
            // 3.其他（包括在同一纵轴线上，即斜率为0）
            for (int j = i + 1; j < points.length; j++) {
                // 所以这两为啥要分开讨论呢？
                // 如果两点的x坐标不同，但y坐标相同，那么斜率是0，两点在一条横轴线上
                // 如果两点的坐标完全相同，即重合，那么无需计算
                // 两点的y坐标相同
                if (points[i][1] == points[j][1]) {
                    same_y++;
                    // 两点的x坐标也相同
                    if (points[i][0] == points[j][0]) {
                        same++;
                    }
                } else {
                    // 两坐标没有相同，计算斜率
                    // 注意这里是dx / dy，这是不可逆的
                    // 试想，A(1,1)、B(1,2)
                    // 这两点的斜率是0，但如果dy / dx，那么斜率是无法计算的，两点在一条纵轴线上
                    // 但这种斜率为0又不可以和上述（两点的x坐标不同，但y坐标相同，那么斜率是0）这种情况混为一谈
                    // 简直离谱，java的double，-0.0和0.0是不同的两个数？？！
                    double dx = points[i][0] - points[j][0], dy = points[i][1] - points[j][1];
                    double rate = dx / dy;
                    if (rate == 0.0) {
                        rate = 0.0;
                    }
                    map.put(rate, map.getOrDefault(rate, 0) + 1);
                }
            }
            max = Math.max(max, same_y);
            for (int val : map.values()) {
                max = Math.max(max, same + val);
            }
            map.clear();
        }
        return max;
    }

    // 多重集合映射
    // 332. 重新安排行程
    // JFK
    // 这题有点坑，这些机票要全部用上才行
    // [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
    // "JFK" "ATL" "SFO" "JFK" "SFO" "ATL"
    public List<String> findItinerary(List<List<String>> tickets) {
        List<String> res = new LinkedList<>();
        // 起始机场：降落机场（优先队列按字符自然排序）
        Map<String, PriorityQueue<String>> map = new HashMap<>();
        for (List<String> ticket : tickets) {
            // computeIfAbsent：若map中还没有这个key，则根据方法给他生成一个value
            PriorityQueue<String> pq = map.computeIfAbsent(ticket.get(0), s -> new PriorityQueue<>());
            pq.offer(ticket.get(1));
        }
        // 有向有环图的深度优先搜索
        dfs(map, res);

        return res;
    }

    // 这道题可以从图的角度思考
    // 题目给定一些点，可以且只可能构成如下几种有向图
    // 1.孤岛图
    // 孤岛即是指，从起始点出发就无法再返回的图
    // 2.环图
    // 环图，顾名思义，就是一个环，跟孤岛对应，从起始点出发后，最终仍能返回起始点
    // 3.孤岛+环图
    // 从起始点出发，有不同路径，可能去往环图，也可能去往孤岛
    // 另外，值得注意的一点是，这是一个具有实际意义的图
    // 只可能存在一个孤岛，试问，一个人如果去了一次孤岛，无法再返回，怎么可能再去往另一个孤岛
    // 所以只可能存在一个孤岛，而且这个孤岛必然是最后去的，而坏是可以存在多个的，因为可以返回
    // 考虑多个环和1个孤岛的情况
    // 可以考虑这种优先策略：先去自然顺序大的环岛，再去自然顺序小的环岛，最后去孤岛
    // 基于只能存在一个孤岛这个思想，再思考点的出入度问题，
    // 先说结论，除了起始点的出度可以比入度大1，其他点的出度和入度必然是相等的
    // 起始点多出的出度，即为最后到达的孤岛
    // 其他的点如果多出出度，那么从起始点到达这个点后，
    // 必然会面临这样的选择，是前往孤岛呢？还是回环？此时不管是做出哪种选择，
    // 都必然导致一些机票用不到，即一些边访问不到
    // 因为，前往孤岛后，必然不能回环，回环后，必然也不能再前往孤岛
    // 即，如果以起始点分隔，环岛可以与环岛结合，环岛和孤岛不能结合
    // 这道题强烈结合现实情境，我觉得可以评成hard。
    public void dfs(Map<String, PriorityQueue<String>> map, List<String> res) {
        Stack<String> stack = new Stack<>();
        stack.push("JFK");
        while (!stack.empty()) {
            PriorityQueue<String> pq;
            // 这一步是点睛之笔
            // 这种干净的代码我啥时候能写出来
            while ((pq = map.get(stack.peek())) != null && pq.size() > 0) {
                stack.push(pq.poll());
            }
            res.add(0, stack.pop());
        }
    }

    // 前缀和 与 积分图

    // 560. 和为K的子数组
    // 注意动态规划的思想
    /*输入:nums = [1,1,1], k = 2
    输出: 2 , [1,1] 与 [1,1] 为两种不同的情况。*/
    public int subarraySum(int[] nums, int k) {
        int res = 0;
        // 前缀和：出现次数
        Map<Integer, Integer> map = new HashMap<>();
        // 坐标以前全要的意思
        map.put(0, 1);
        int curSum = 0;
        for (int num : nums) {
            // 计算当前前缀
            curSum += num;
            res += map.getOrDefault(curSum - k, 0);
            // 存储前缀和
            map.put(curSum, map.getOrDefault(curSum, 0) + 1);
        }
        return res;
    }

    // 练习
    // 566. 重塑矩阵
    public int[][] matrixReshape(int[][] nums, int r, int c) {
        int row = nums.length;
        if (row == 0) {
            return nums;
        }
        int col = nums[0].length;
        if (row * col != r * c) {
            return nums;
        }
        int[][] res = new int[r][c];
        int count = 0;
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                res[i][j] = nums[count / col][count % col];
                count++;
            }
        }

        return res;
    }

    // 503. 下一个更大元素 II
    public int[] nextGreaterElements(int[] nums) {
        int[] res = new int[nums.length];
        Arrays.fill(res, -1);
        // 单调栈-保持递减
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < 2 * nums.length; i++) {
            int index = i % nums.length;
            while (!stack.empty()) {
                if (nums[index] <= nums[stack.peek()]) {
                    break;
                }
                res[stack.pop()] = nums[index];
            }
            stack.push(index);
        }

        return res;
    }

    // 217. 存在重复元素
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            if (set.contains(num)) {
                return true;
            }
            set.add(num);
        }
        return false;
    }

    // 697. 数组的度
    public int findShortestSubArray(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        // 记录各数字第一次出现的位置和出现的次数
        Map<Integer, int[]> map = new HashMap<>();
        int maxD = 1; // 目前最大的度
        int minZ = 1; // 目前最小的值
        for (int i = 0; i < nums.length; i++) {
            int[] log;
            if (map.containsKey(nums[i])) {
                log = map.get(nums[i]);
                log[1] += 1;
                if (log[1] >= maxD) {
                    minZ = log[1] > maxD ? i - log[0] + 1 : Math.min(minZ, i - log[0] + 1);
                    maxD = log[1];
                }
            } else {
                log = new int[2];
                log[0] = i;
                log[1] = 1;
            }
            map.put(nums[i], log);
        }
        return minZ;
    }

    // 594. 最长和谐子序列
    public int findLHS(int[] nums) {
        int res = 0;
        // 数字：出现次数
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
            if (map.containsKey(num - 1)) {
                res = Math.max(res, map.get(num) + map.get(num - 1));
            }
            if (map.containsKey(num + 1)) {
                res = Math.max(res, map.get(num) + map.get(num + 1));
            }
        }
        return res;
    }

    // 287. 寻找重复数
    // 那个快慢指针有点不好理解
    // 只好看看二分查找的解法
    public int findDuplicate(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = left + right >> 1;
            int count = 0;
            for (int num : nums) {
                if (num <= mid) {
                    count++;
                }
            }
            if (count > mid) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    // 263. 丑数
    public boolean isUgly(int n) {
        // 2\3\5
        if (n < 1) {
            return false;
        }
        while (n % 2 == 0) {
            n /= 2;
        }
        while (n % 3 == 0) {
            n /= 3;
        }
        while (n % 5 == 0) {
            n /= 5;
        }
        return n == 1;
    }

    // 264. 丑数 II
    // 2\3\5
    public int nthUglyNumber(int n) {
        // 2、3、5
        int[] dp = new int[n];
        dp[0] = 1;
        int[] indexes = new int[3];

        for (int i = 1; i < n; i++) {
            int num2 = 2 * dp[indexes[0]];
            int num3 = 3 * dp[indexes[1]];
            int num5 = 5 * dp[indexes[2]];
            dp[i] = Math.min(num2, Math.min(num3, num5));
            if (num2 == dp[i]) {
                indexes[0]++;
            }
            if (num3 == dp[i]) {
                indexes[1]++;
            }
            if (num5 == dp[i]) {
                indexes[2]++;
            }
        }

        return dp[n - 1];
    }

    // 313. 超级丑数
    // 丑数的生成即是不断拿原有的丑数乘已经生成的丑数
    // 初始的丑数是1，考虑第一轮最小的丑数即为数组中的最小的数
    // 第一轮可以理解为在原有数组的基础上，求出相乘数为1，也就是数本身可生成的丑数
    // 第二轮可以理解为，相乘数为2时，可生成的丑数
    // 可以想见，第一轮求出的丑数，在经过小顶堆的洗礼，其堆顶必然也是完整丑数的最小值，姑且称之第一位
    // 而到第二轮，可以知道的是，在所有相乘数为2的丑数中，最小的必然是最小数乘最小数，
    // 但堆中还留有第一轮的丑数结果，无法敲定相乘数为2的最小丑数在相乘数为1的丑数(这里必然不包含最小数本身)中的排序
    // 但我们有了小顶堆
    // tmd，还有溢出问题
    // 以上注释先留着，但下面会换个基于此的优化解法，上面那个太慢了
    public int nthSuperUglyNumber(int n, int[] primes) {
        int[] dp = new int[n];
        dp[0] = 1;

        PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(o -> o[0] * dp[o[1]]));
        for (int prime : primes) {
            pq.offer(new int[]{prime, 0});
        }
        for (int i = 1; i < n; i++) {
            int[] cur = pq.poll();
            dp[i] = cur[0] * dp[cur[1]];
            int[] tmp;
            while ((tmp = pq.peek()) != null && tmp[0] * dp[tmp[1]] == dp[i]) {
                pq.poll();
                tmp[1] += 1;
                pq.offer(tmp);
            }
            cur[1] += 1;
            pq.offer(cur);
        }
        return dp[n - 1];
    }

    // 870. 优势洗牌
    public int[] advantageCount(int[] A, int[] B) {
        int len = B.length;
        Arrays.sort(A);
        Queue<Integer> aq = new ArrayDeque<>();
        List<int[]> list = new ArrayList<>();
        for (int i = 0; i < len; i++) {
            aq.offer(A[i]);
            list.add(new int[]{B[i], i});
        }
        list.sort(Comparator.comparingInt(o -> o[0]));
        Queue<Integer> tmp = new ArrayDeque<>();
        for(int[] arr:list){
            while (!aq.isEmpty() && arr[0] >= aq.element()) {
                tmp.add(aq.poll());
            }
            A[arr[1]] = aq.isEmpty() ? tmp.remove() : aq.remove();
        }

        return A;
    }

    public static void main(String[] args) {
        Structure structure = new Structure();

        // MyQueue myQueue = new MyQueue();
        // myQueue.push(1);

        // 输入：lists = [[1,4,5],[1,3,4],[2,6]]
        // 输出：[1,1,2,3,4,4,5,6]
        /*ListNode node00 = new ListNode(1);
        ListNode node01 = new ListNode(4);
        node00.next = node01;
        node01.next = new ListNode(5);

        ListNode node10 = new ListNode(1);
        ListNode node11 = new ListNode(3);
        node10.next = node11;
        node11.next = new ListNode(4);

        ListNode node21 = new ListNode(2);
        node21.next = new ListNode(6);

        ListNode[] lists = new ListNode[]{node00, node10, node21};*/
        // System.out.println(structure.mergeKLists(lists));
        // int[] nums = new int[]{1, 3, -1, -3, 5, 3, 6, 7};
        // System.out.println(Arrays.toString(structure.maxSlidingWindow(nums, 3)));

        // [[0,1],[0,0],[0,4],[0,-2],[0,-1],[0,3],[0,-4]]
        int[][] points = new int[][]{
                {0, 1},
                {0, 0},
                {0, 4},
                {0, -2},
                {0, -1},
                {0, 3},
                {0, -4}
        };
        // System.out.println(structure.maxPoints(points));

        // [["JFK","KUL"],["JFK","NRT"],["NRT","JFK"]]
        /*List<List<String>> tickets = new ArrayList<>();
        List<String> t1 = new ArrayList<>();
        t1.add("JFK");
        t1.add("B");
        tickets.add(t1);
        List<String> t2 = new ArrayList<>();
        t2.add("JFK");
        t2.add("C");
        tickets.add(t2);
        List<String> t3 = new ArrayList<>();
        t3.add("C");
        t3.add("JFK");
        tickets.add(t3);
        List<String> t4 = new ArrayList<>();
        t4.add("JFK");
        t4.add("A");
        tickets.add(t4);
        List<String> t5 = new ArrayList<>();
        t5.add("A");
        t5.add("JFK");
        tickets.add(t5);*/
        // System.out.println(structure.findItinerary(tickets));

        int[] nums = new int[]{0, 0, 0};
        // System.out.println(structure.subarraySum(nums, 0));

        MyStack myStack = new MyStack();
        myStack.push(1);
        myStack.push(2);
        myStack.pop();
        // System.out.println(myStack.top());

        int[] primes = new int[]{2, 7, 13, 19};
        // System.out.println(structure.nthSuperUglyNumber(8, primes));

        // System.out.println(structure.nthUglyNumber(10));

        int[] A = new int[]{12, 24, 8, 32};
        int[] B = new int[]{13, 25, 32, 11};
        System.out.println(Arrays.toString(structure.advantageCount(A, B)));
    }
}

// 232. 用栈实现队列
// 使用两个栈实现先入先出队列
class MyQueue {
    Stack<Integer> in, out;

    /**
     * Initialize your data structure here.
     */
    public MyQueue() {
        in = new Stack<>();
        out = new Stack<>();
    }

    /**
     * Push element x to the back of queue.
     */
    public void push(int x) {
        in.push(x);
    }

    /**
     * Removes the element from in front of queue and returns that element.
     */
    public int pop() {
        if (out.empty()) {
            in2out();
        }
        return out.pop();
    }

    /**
     * Get the front element.
     */
    public int peek() {
        if (out.empty()) {
            in2out();
        }
        return out.peek();
    }

    /**
     * Returns whether the queue is empty.
     */
    public boolean empty() {
        return in.empty() && out.empty();
    }

    void in2out() {
        while (!in.empty()) {
            out.push(in.pop());
        }
    }
}

// 155. 最小栈
class MinStack {
    Stack<Integer> stack, min;

    /**
     * initialize your data structure here.
     */
    public MinStack() {
        stack = new Stack<>();
        min = new Stack<>();
    }

    public void push(int val) {
        stack.push(val);
        if (min.empty() || val <= min.peek()) {
            min.push(val);
        }
    }

    public void pop() {
        if (min.peek().equals(stack.pop())) {
            min.pop();
        }
    }

    public int top() {
        return stack.peek();
    }

    public int getMin() {
        return min.peek();
    }
}

class PQueue {
    List<Integer> list = new ArrayList<>();

    // 获取最大值(不删除)
    int top() {
        return list.get(0);
    }

    // 插入任意值：把新的数字放在最后一位，然后上浮
    void push(int k) {
        list.add(k);
        swim(list.size() - 1);
    }

    // 删除最大值：把最后一个数字挪到开头，然后下沉
    int pop() {
        int top = list.remove(0);
        list.add(0, list.get(list.size() - 1));
        return top;
    }

    // 上浮
    void swim(int pos) {
        while (pos > 1 && list.get(pos / 2) < list.get(pos)) {
            Collections.swap(list, pos, pos / 2);
            pos = pos / 2;
        }
    }

    // 下沉
    void sink(int pos) {
        while (pos * 2 < list.size()) {
            int index = pos * 2;
            if (index < list.size() && list.get(index) < list.get(index + 1)) {
                index++;
            }
            if (list.get(index) <= list.get(pos)) {
                break;
            }
            Collections.swap(list, pos, index);
            pos = index;
        }
    }
}

// 303. 区域和检索 - 数组不可变
// 一维 前缀和
class NumArray {
    int[] sums;

    public NumArray(int[] nums) {
        sums = new int[nums.length + 1];
        for (int i = 1; i < sums.length; i++) {
            sums[i] = nums[i - 1] + sums[i - 1];
        }
    }

    public int sumRange(int left, int right) {
        return sums[right + 1] - sums[left];
    }
}

// 304. 二维区域和检索 - 矩阵不可变
// 二维 积分图
class NumMatrix {
    int[][] sums;

    public NumMatrix(int[][] matrix) {
        sums = new int[matrix.length + 1][matrix[0].length + 1];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                sums[i + 1][j + 1] = sums[i][j + 1] + sums[i + 1][j] - sums[i][j] + matrix[i][j];
            }
        }
    }

    public int sumRegion(int row1, int col1, int row2, int col2) {
        return sums[row2 + 1][col2 + 1] - sums[row2 + 1][col1] - sums[row1][col2 + 1] + sums[row1][col1];
    }
}

// 225. 用队列实现栈
class MyStack {
    Queue<Integer> myStack;

    /**
     * Initialize your data structure here.
     */
    public MyStack() {
        myStack = new LinkedList<>();
    }

    /**
     * Push element x onto stack.
     */
    public void push(int x) {
        int size = myStack.size();
        myStack.offer(x);
        while (size > 0) {
            myStack.offer(myStack.remove());
            size--;
        }
    }

    /**
     * Removes the element on top of the stack and returns that element.
     */
    public int pop() {
        return myStack.remove();
    }

    /**
     * Get the top element.
     */
    public int top() {
        return myStack.element();
    }

    /**
     * Returns whether the stack is empty.
     */
    public boolean empty() {
        return myStack.isEmpty();
    }
}

// 307. 区域和检索 - 数组可修改
class NumArray2 {

    public NumArray2(int[] nums) {

    }

    public void update(int index, int val) {

    }

    public int sumRange(int left, int right) {
        return 1;
    }
}
