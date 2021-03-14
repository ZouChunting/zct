package leetcode;

import com.sun.xml.internal.xsom.XSTerm;
import javafx.util.Pair;

import java.lang.reflect.Parameter;
import java.util.*;

public class Search {
    int[] arr = new int[]{-1, 0, 1, 0, -1};

    void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

    // 深度优先搜索
    // 695. 岛屿的最大面积
    public int maxAreaOfIsland(int[][] grid) {
        int area = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    grid[i][j] = 0;
                    int cur_area = 1;
                    Stack<Pair<Integer, Integer>> stack = new Stack<>();
                    stack.push(new Pair<>(i, j));
                    while (!stack.empty()) {
                        Pair<Integer, Integer> pair = stack.pop();
                        int r = pair.getKey();
                        int c = pair.getValue();
                        for (int k = 0; k < 4; k++) {
                            int x = r + arr[k];
                            int y = c + arr[k + 1];
                            if (x >= 0 && y >= 0 && x < grid.length && y < grid[0].length && grid[x][y] == 1) {
                                grid[x][y] = 0;
                                cur_area++;
                                stack.push(new Pair<>(x, y));
                            }
                        }
                    }
                    area = Math.max(area, cur_area);
                }
            }
        }

        return area;
    }

    public int maxAreaOfIsland2(int[][] grid) {
        int area = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == 1) {
                    area = Math.max(area, dfs(grid, i, j));
                }
            }
        }

        return area;
    }

    // 深度优先
    int dfs(int[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] == 0) {
            return 0;
        }
        // 用过就全部置0
        grid[i][j] = 0;
        int count = 1;
        count += dfs(grid, i + 1, j);
        count += dfs(grid, i, j + 1);
        count += dfs(grid, i - 1, j);
        count += dfs(grid, i, j - 1);
        return count;
    }

    // 547. 省份数量
    public int findCircleNum(int[][] isConnected) {
        int num = 0;
        Set<Integer> visited = new HashSet<>();
        for (int i = 0; i < isConnected.length; i++) {
            if (!visited.contains(i)) {
                visited.add(i);
                dfs(isConnected, i, visited);
                num++;
            }
        }
        return num;
    }

    void dfs(int[][] isConnected, int i, Set<Integer> visited) {
        for (int j = 0; j < isConnected.length; j++) {
            if (!visited.contains(j) && isConnected[i][j] == 1) {
                visited.add(j);
                dfs(isConnected, j, visited);
            }
        }
    }

    // 417. 太平洋大西洋水流问题
    public List<List<Integer>> pacificAtlantic(int[][] matrix) {
        List<List<Integer>> res = new ArrayList<>();
        int row = matrix.length;
        if (row == 0) {
            return res;
        }
        int col = matrix[0].length;
        // 可流向太平洋（左 上）
        boolean[][] canReachP = new boolean[row][col];
        // 可流向大西洋（右 下）
        boolean[][] canReachA = new boolean[row][col];

        for (int i = 0; i < row; i++) {
            pour(matrix, canReachP, i, 0);
            pour(matrix, canReachA, i, col - 1);
        }

        for (int j = 0; j < col; j++) {
            pour(matrix, canReachP, 0, j);
            pour(matrix, canReachA, row - 1, j);
        }

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (canReachP[i][j] && canReachA[i][j]) {
                    List<Integer> tmp = new ArrayList<>();
                    tmp.add(i);
                    tmp.add(j);
                    res.add(tmp);
                }
            }
        }
        return res;
    }

    void pour(int[][] matrix, boolean[][] canReach, int row, int col) {
        if (canReach[row][col]) {
            return;
        }
        canReach[row][col] = true;
        for (int i = 0; i < 4; i++) {
            int x = row + arr[i];
            int y = col + arr[i + 1];
            if (x >= 0 && x < matrix.length && y >= 0 && y < matrix[0].length && matrix[row][col] <= matrix[x][y]) {
                pour(matrix, canReach, x, y);
            }
        }
    }

    // 回溯
    // 46. 全排列
    // 数组内不重复是很重要的条件
    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums.length == 0) {
            return res;
        }

        List<Integer> list = new ArrayList<>();
        for (int num : nums) {
            list.add(num);
        }
        backTracking(list, 0, res);

        return res;
    }

    void backTracking(List<Integer> list, int depth, List<List<Integer>> res) {
        if (depth == list.size() - 1) {
            // 存进res
            // 在 Java 中，参数传递是 值传递，对象类型变量在传参的过程中，复制的是变量的地址。
            // 这些地址被添加到 res 变量，但实际上指向的是同一块内存地址。
            // 若直接res.add(list);最后会得到全部相同的（1，2，3）6个结果
            res.add(new ArrayList<>(list));
            return;
        }
        // 此处从depth开始，是为(1,2,3)不变动也是全排列的一种
        for (int i = depth; i < list.size(); i++) {
            Collections.swap(list, depth, i);
            backTracking(list, depth + 1, res);
            Collections.swap(list, depth, i);
        }
    }

    // 77. 组合
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> tmp = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            tmp.add(0);
        }
        backTracking(n, k, 1, 0, tmp, res);
        return res;
    }

    void backTracking(int n, int k, int index, int count, List<Integer> tmp, List<List<Integer>> res) {
        if (count == k) {
            res.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = index; i <= n; i++) {
            tmp.set(count++, i);
            backTracking(n, k, i + 1, count, tmp, res);
            count--;
        }
    }

    // 79. 单词搜索
    public boolean exist(char[][] board, String word) {
        if (board.length == 0 || board[0].length == 0) {
            return false;
        }
        boolean[][] visited = new boolean[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                boolean res = backTracking(i, j, board, 0, word, visited);
                if (res) {
                    return true;
                }
            }
        }
        return false;
    }

    // i,j 字符组检索坐标 board原字符组 index字符串检索位置 word字符串 visited字符组访问标记
    boolean backTracking(int i, int j, char[][] board, int index, String word, boolean[][] visited) {
        if (board[i][j] != word.charAt(index)) {
            return false;
        }
        if (index == word.length() - 1) {
            return true;
        }

        boolean res = false;

        visited[i][j] = true;
        for (int k = 0; k < 4; k++) {
            int newi = i + arr[k];
            int newj = j + arr[k + 1];
            if (newi >= 0 && newj >= 0 && newi <= board.length - 1 && newj <= board[0].length - 1) {
                if (!visited[newi][newj]) {
                    boolean flag = backTracking(newi, newj, board, index + 1, word, visited);
                    if (flag) {
                        res = true;
                        break;
                    }
                }
            }
        }
        visited[i][j] = false;

        return res;
    }

    // 51. N 皇后
    // 这道题仍需好好看看，因为对递归理解不透彻，失败了很多次
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> res = new ArrayList<>();
        char[][] chessboard = new char[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                chessboard[i][j] = '.';
            }
        }
        backTracking(chessboard, res, 0);
        return res;
    }

    void backTracking(char[][] chessboard, List<List<String>> res, int i) {
        if (i == chessboard.length) {
            List<String> tmp = new ArrayList<>();
            for (char[] row : chessboard) {
                tmp.add(new String(row));
            }
            res.add(tmp);
            return;
        }
        // 根据条件，每行只有一个数
        for (int j = 0; j < chessboard.length; j++) {
            if (isChess(i, j, chessboard)) {
                chessboard[i][j] = 'Q';
                backTracking(chessboard, res, i + 1);
                chessboard[i][j] = '.';
            }
        }
    }

    boolean isChess(int i, int j, char[][] chessboard) {
        // 判断上面有没有皇后
        for (int m = 0; m < i; m++) {
            if (chessboard[m][j] == 'Q') {
                return false;
            }
        }
        // 判断左上角有没有皇后
        for (int m = i - 1, n = j - 1; m >= 0 && n >= 0; m--, n--) {
            if (chessboard[m][n] == 'Q') {
                return false;
            }
        }
        // 判断右上角有没有皇后
        for (int m = i - 1, n = j + 1; m >= 0 && n < chessboard.length; m--, n++) {
            if (chessboard[m][n] == 'Q') {
                return false;
            }
        }

        return true;
    }

    // 广度优先搜索
    // 934. 最短的桥
    public int shortestBridge(int[][] A) {
        boolean[][] visited = new boolean[A.length][A[0].length];
        Queue<int[]> points = new LinkedList<>();
        boolean flag = false;
        // 深度优先寻找第一个岛
        for (int i = 0; i < A.length; i++) {
            if (flag) {
                break;
            }
            for (int j = 0; j < A[0].length; j++) {
                if (A[i][j] == 1) {
                    dfs(A, visited, points, i, j);
                    flag = true;
                    break;
                }
            }
        }
        int res = -1;
        // 广度优先寻找第二个岛
        while (!points.isEmpty()) {
            res++;
            int size = points.size();
            for (int i = 0; i < size; i++) {
                int[] point = points.poll();
                for (int k = 0; k < 4; k++) {
                    int row = point[0] + arr[k];
                    int col = point[1] + arr[k + 1];
                    if (row < 0 || col < 0 || row >= A.length || col >= A[0].length || visited[row][col]) {
                        continue;
                    }
                    if (A[row][col] == 1) {
                        return res;
                    }
                    visited[row][col] = true;
                    points.add(new int[]{row, col});
                }
            }
        }
        return res;
    }

    void dfs(int[][] A, boolean[][] visited, Queue<int[]> points, int i, int j) {
        if (i < 0 || j < 0 || i >= A.length || j >= A[0].length || visited[i][j] || A[i][j] == 0) {
            return;
        }
        visited[i][j] = true;
        points.add(new int[]{i, j});
        dfs(A, visited, points, i - 1, j);
        dfs(A, visited, points, i + 1, j);
        dfs(A, visited, points, i, j - 1);
        dfs(A, visited, points, i, j + 1);
    }

    // 126. 单词接龙 II
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        List<List<String>> res = new ArrayList<>();
        if (wordList.size() == 0 || !wordList.contains(endWord)) {
            return res;
        }
        Set<String> wordSet = new HashSet<>(wordList);

        return res;
    }

    void bidirectionalBFS(String beginWord, String endWord, Set<String> wordSet) {
        // 记录访问过的单词
        Set<String> visited = new HashSet<>();
        visited.add(beginWord);
        visited.add(endWord);

        Set<String> beginVisited = new HashSet<>();
        beginVisited.add(beginWord);
        Set<String> endVisited = new HashSet<>();
        endVisited.add(endWord);

        while (!beginVisited.isEmpty()) {
            if (beginVisited.size() > endVisited.size()) {

            }
        }

    }

    // 130. 被围绕的区域
    public void solve(char[][] board) {
        for (int i = 0; i < board.length; i++) {
            if (board[i][0] == 'O') {
                dfs(board, i, 0);
            }
            if (board[i][board[0].length - 1] == 'O') {
                dfs(board, i, board[0].length - 1);
            }
        }
        for (int j = 0; j < board[0].length; j++) {
            if (board[0][j] == 'O') {
                dfs(board, 0, j);
            }
            if (board[board.length - 1][j] == 'O') {
                dfs(board, board.length - 1, j);
            }
        }
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (board[i][j] == '-') {
                    board[i][j] = 'O';
                } else {
                    board[i][j] = 'X';
                }
            }
        }
    }

    void dfs(char[][] board, int i, int j) {
        if (i < 0 || j < 0 || i >= board.length || j >= board[0].length || board[i][j] != 'O') {
            return;
        }
        board[i][j] = '-';
        for (int k = 0; k < 4; k++) {
            int row = i + arr[k];
            int col = j + arr[k + 1];
            dfs(board, row, col);
        }
    }

    // 257. 二叉树的所有路径
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> res = new ArrayList<>();
        if (root != null) {
            dfs(root, res, root.val + "");
        }
        return res;
    }

    void dfs(TreeNode root, List<String> res, String tmp) {
        // 到达叶节点
        if (root.left == null && root.right == null) {
            res.add(tmp);
            return;
        }
        if (root.left != null) {
            dfs(root.left, res, tmp + "->" + root.left.val);
        }
        if (root.right != null) {
            dfs(root.right, res, tmp + "->" + root.right.val);
        }
    }

    // 47. 全排列 II
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums.length == 0) {
            return res;
        }
        Arrays.sort(nums);
        Deque<Integer> tmp = new ArrayDeque<>();
        boolean[] used = new boolean[nums.length];
        backTracking(nums, tmp, used, res, 0);
        return res;
    }

    void backTracking(int[] nums, Deque<Integer> tmp, boolean[] used, List<List<Integer>> res, int index) {
        if (index == nums.length) {
            res.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) {
                continue;
            }
            if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) {
                continue;
            }
            used[i] = true;
            tmp.addLast(nums[i]);
            backTracking(nums, tmp, used, res, index + 1);
            tmp.removeLast();
            used[i] = false;
        }
    }

    // 40. 组合总和 II
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (candidates.length == 0) {
            return res;
        }
        Arrays.sort(candidates);
        Deque<Integer> tmp = new ArrayDeque<>();
        boolean[] used = new boolean[candidates.length];
        dfs(candidates, target, 0, 0, tmp, res, used);

        return res;
    }

    // [1, 1, 2, 5, 6, 7, 10]
    void dfs(int[] candidates, int target, int index, int sum, Deque<Integer> tmp, List<List<Integer>> res, boolean[] used) {
        if (sum > target) {
            return;
        }
        if (sum == target) {
            res.add(new ArrayList<>(tmp));
            return;
        }

        for (int i = index; i < candidates.length; i++) {
            if (i > 0 && candidates[i] == candidates[i - 1] && !used[i - 1]) {
                continue;
            }
            tmp.add(candidates[i]);
            used[i] = true;
            dfs(candidates, target, i + 1, sum + candidates[i], tmp, res, used);
            used[i] = false;
            tmp.removeLast();
        }
    }

    // 37. 解数独
    public void solveSudoku(char[][] board) {

    }

    // 310. 最小高度树
    // 我愿称之为萎缩法，一层一层剪去所有叶子
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        List<Integer> res = new ArrayList<>();
        if (n <= 0) {
            return res;
        }
        if (n == 1) {
            res.add(0);
            return res;
        }
        int[] degree = new int[n];
        List<List<Integer>> tree = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            tree.add(new ArrayList<>());
        }
        for (int[] edge : edges) {
            degree[edge[0]]++;
            degree[edge[1]]++;
            tree.get(edge[0]).add(edge[1]);
            tree.get(edge[1]).add(edge[0]);
        }
        Deque<Integer> leaves = new ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            if (degree[i] == 1) {
                leaves.offer(i);
            }
        }
        while (!leaves.isEmpty()) {
            res = new ArrayList<>();
            int size = leaves.size();
            for (int i = 0; i < size; i++) {
                int leaf = leaves.remove();
                res.add(leaf);
                List<Integer> neighbors = tree.get(leaf);
                for (int neighbor : neighbors) {
                    degree[neighbor]--;
                    if (degree[neighbor] == 1) {
                        leaves.offer(neighbor);
                    }
                }
            }
        }

        return res;
    }

    public static void main(String[] args) {
        Search search = new Search();

        // [[1,1,0,0,0],[1,1,0,0,0],[0,0,0,1,1],[0,0,0,1,1]]
        int[][] grid = new int[][]{
                {1, 1, 0, 0, 0},
                {1, 1, 0, 0, 0},
                {0, 0, 0, 1, 1},
                {0, 0, 0, 1, 1}
        };
        // System.out.println(search.maxAreaOfIsland(grid));

        int[][] matrix = new int[0][0];

        int[] nums = {1, 2, 1};
        // System.out.println(search.permute(nums));

        // int n = 4, k = 2;
        // System.out.println(search.combine(n, k));

        char[][] board = {{'A', 'B', 'C', 'E'}, {'S', 'F', 'C', 'S'}, {'A', 'D', 'E', 'E'}};
        String word = "SEE";
        // System.out.println(search.exist(board, word));

        int n = 4;
        // System.out.println(search.solveNQueens(n));
        int[][] A = new int[][]{
                {1, 1, 1, 1, 1},
                {1, 0, 0, 0, 1},
                {1, 0, 1, 0, 1},
                {1, 0, 0, 0, 1},
                {1, 1, 1, 1, 1}
        };
        // System.out.println(search.shortestBridge(A));

        TreeNode node5 = new TreeNode(5);
        TreeNode node3 = new TreeNode(3);
        TreeNode node2 = new TreeNode(2, null, node5);
        TreeNode root = new TreeNode(1, node2, node3);
        // System.out.println(search.binaryTreePaths(root));

        // System.out.println(search.permuteUnique(nums));

        int[] candidates = new int[]{2, 5, 2, 1, 2};
        int target = 5;
        System.out.println(search.combinationSum2(candidates, target));
    }
}

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;

    TreeNode() {
    }

    TreeNode(int val) {
        this.val = val;
    }

    TreeNode(int val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}
/*
1 1 2

 */