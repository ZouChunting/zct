package leetcode;

import java.util.*;

public class Tree {
    // 树
    // 104. 二叉树的最大深度
    public int maxDepth(TreeNode root) {
        return root == null ? 0 : 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }

    // 110. 平衡二叉树
    // 涉及到递归我就一脸懵
    // 理解时可以画一棵树从下至上计算理解
    public boolean isBalanced(TreeNode root) {
        return calDepth(root) != -1;
    }

    int calDepth(TreeNode node) {
        if (node == null) {
            return 0;
        }
        // 各自计算左右深度
        int left = calDepth(node.left), right = calDepth(node.right);

        if (left == -1 || right == -1 || Math.abs(left - right) > 1) {
            return -1;
        }
        // 取最大深度作为
        return 1 + Math.max(left, right);
    }

    // 543. 二叉树的直径
    int maxDiam = 0;

    public int diameterOfBinaryTree(TreeNode root) {
        if (root == null) {
            return 0;
        }
        calLen(root);
        return maxDiam;
    }

    int calLen(TreeNode node) {
        if (node.left == null && node.right == null) {
            return 0;
        }
        int left = node.left == null ? 0 : 1 + calLen(node.left);
        int right = node.right == null ? 0 : 1 + calLen(node.right);
        maxDiam = Math.max(maxDiam, left + right);
        return Math.max(left, right);
    }

    // 437. 路径总和 III
    // 解决本次要掌握前缀和思想、递归思想和逐层计算的思想
    int sumP = 0;

    public int pathSum(TreeNode root, int targetSum) {
        // 前缀和
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, 1);
        calPath(root, targetSum, 0, map);
        return sumP;
    }

    // 已知，前缀和用法
    // 一数组中有坐标i,j，j>i
    // 若欲知i与j之间数的和
    // 则将前缀和j减去i
    void calPath(TreeNode node, int targetSum, int curSum, Map<Integer, Integer> map) {
        if (node == null) {
            return;
        }
        curSum += node.val;
        // 这里碰到一个错误，
        // 原本写的是targetSum - curSum
        // 后来改为curSum - targetSum
        // 考虑上述前缀和用法
        // map中存放的此节点以前的所有前缀和
        // curSum为当前节点的前缀和
        // curSum - targetSum即为计算区间和，
        // 若能在map中找到curSum - targetSum的值，
        // 说明前面有某一前缀和，与当前前缀和构成的区间和，恰为目标值
        // 这实际上是一种自下而上的想法
        // 而targetSum - curSum，其实是在计算，当前这个和加完，能不能就达到目标值了呢
        // 这是一种自上而下的想法，跟前缀和完全不是一个思路
        sumP += map.getOrDefault(curSum - targetSum, 0);
        // 存入前缀和
        map.put(curSum, map.getOrDefault(curSum, 0) + 1);
        calPath(node.left, targetSum, curSum, map);
        calPath(node.right, targetSum, curSum, map);
        // 这里必须在把curSum去掉的原因
        // 虽然说递归是一种线性计算
        // 但是我们仍可以按照层来思考，假设每一层的点都是并发计算的
        // 考虑案例 根1左-1右-3目标-1
        // 执行流程如下，先计算当前前缀和1，比对目标值并存入前缀和
        // 接下来分流的，分别计算左-1和右-3，本来按并发的逻辑，而且左右各自存自己的前缀和，是没有问题的
        // 然而因为递归的线性计算，而且前缀和是传址的
        // 所以先计算的左节点，必然会把自己的前缀和传递给后计算的右节点
        // 因此需要回溯，左节点计算完之后，把自己的前缀和去掉，把一个只有父节点的前缀和的干净的前缀和给有节点计算
        // 而纵览整个递归计算过程，实际上是现在最左节点前缀和全部计算完毕，到最底层，再逐层往上计算并回溯旧值
        map.put(curSum, map.get(curSum) - 1);
    }

    // 101. 对称二叉树
    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }

        return mirror(root.left, root.right);
    }

    boolean mirror(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if (left == null || right == null || left.val != right.val) {
            return false;
        }

        return mirror(left.left, right.right) && mirror(left.right, right.left);
    }

    // 1110. 删点成林
    public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {
        List<TreeNode> res = new ArrayList<>();
        // 将待删除的点放入set集合，方便寻找
        Set<Integer> set = new HashSet<>();
        for (int del : to_delete) {
            set.add(del);
        }
        root = forest(root, set, res);
        if (root != null) {
            res.add(root);
        }
        return res;
    }

    TreeNode forest(TreeNode node, Set<Integer> set, List<TreeNode> res) {
        if (node == null) {
            return null;
        }
        node.left = forest(node.left, set, res);
        node.right = forest(node.right, set, res);
        // 若该点命中需删除点
        if (set.contains(node.val)) {
            if (node.left != null) {
                res.add(node.left);
            }
            if (node.right != null) {
                res.add(node.right);
            }
            node = null;
        }
        return node;
    }

    // 层次遍历

    // 637. 二叉树的层平均值
    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int count = queue.size();
            double sum = 0;
            for (int i = 0; i < count; i++) {
                TreeNode tmp = queue.remove();
                sum += tmp.val;
                if (tmp.left != null) {
                    queue.add(tmp.left);
                }
                if (tmp.right != null) {
                    queue.add(tmp.right);
                }
            }
            res.add(sum / count);
        }

        return res;
    }

    // 前中后序遍历

    // 105. 从前序与中序遍历序列构造二叉树
    // 没有重复的数字很重要，这代表我们可以用map
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            map.put(inorder[i], i);
        }

        return build(preorder, map, 0, 0, inorder.length - 1);
    }

    // 前序、中序
    // int[] left = Arrays.copyOfRange(inorder, 0, index);
    // 复制数组很消耗资源
    TreeNode build(int[] preorder, Map<Integer, Integer> inorder, int leftPre, int leftIn, int rightIn) {
        if (leftIn > rightIn) {
            return null;
        }
        TreeNode node = new TreeNode();
        node.val = preorder[leftPre];
        int index = inorder.getOrDefault(preorder[leftPre], rightIn);
        node.left = build(preorder, inorder, leftPre + 1, leftIn, index - 1);
        node.right = build(preorder, inorder, leftPre + 1 + (index - leftIn), index + 1, rightIn);
        return node;
    }

    // 144. 二叉树的前序遍历
    // 注意：这是深度优先
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.add(root);
        while (!stack.empty()) {
            TreeNode node = stack.pop();
            res.add(node.val);
            if (node.right != null) {
                stack.add(node.right);
            }
            if (node.left != null) {
                stack.add(node.left);
            }
        }

        return res;
    }

    // 二叉查找树
    // 题目要点：两点交换、结构不变
    // 中序遍历：左根右
    // 刷hard题的时候明显感觉自己水平不够，除了有时题目不懂之外，理解答案也很困难
    // 所以变换一下策略，先跳过hard题，以刷完所有知识点为主，然后再返回过来挑战新难度
    public void recoverTree(TreeNode root) {
        if (root == null) {
            return;
        }
        Deque<TreeNode> deque = new ArrayDeque<>();
        deque.add(root);
        while (!deque.isEmpty()) {

        }
    }

    List<TreeNode> nodes = new ArrayList<>();

    void midOrder(TreeNode node) {
        if (node.left == null) {

        }
    }

    // 669. 修剪二叉搜索树
    // 注意重点：这是二叉查找树，有序的！！！
    public TreeNode trimBST(TreeNode root, int low, int high) {
        if (root == null) {
            return root;
        }

        if (root.val < low) {
            return trimBST(root.right, low, high);
        }

        if (root.val > high) {
            return trimBST(root.left, low, high);
        }

        root.left = trimBST(root.left, low, high);
        root.right = trimBST(root.right, low, high);

        return root;
    }

    // 226. 翻转二叉树
    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode left = root.left, right = root.right;
        root.left = invertTree(right);
        root.right = invertTree(left);
        return root;
    }

    // 617. 合并二叉树
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null) {
            return root2;
        }
        if (root2 == null) {
            return root1;
        }
        root1.val += root2.val;
        root1.left = mergeTrees(root1.left, root2.left);
        root1.right = mergeTrees(root1.right, root2.right);

        return root1;
    }

    // 572. 另一个树的子树
    public boolean isSubtree(TreeNode root, TreeNode subRoot) {
        if (root == null && subRoot == null) {
            return true;
        }
        if (root == null || subRoot == null) {
            return false;
        }
        return isSame(root, subRoot)
                || isSubtree(root.left, subRoot)
                || isSubtree(root.right, subRoot);
    }

    boolean isSame(TreeNode root, TreeNode subRoot) {
        if (root == null && subRoot == null) {
            return true;
        }

        return root != null && subRoot != null
                && root.val == subRoot.val
                && isSame(root.left, subRoot.left)
                && isSame(root.right, subRoot.right);
    }

    // 404. 左叶子之和
    int sum = 0;

    public int sumOfLeftLeaves(TreeNode root) {
        plusLeft(root, false);
        return sum;
    }

    void plusLeft(TreeNode root, boolean isLeft) {
        if (root == null) {
            return;
        }
        if (isLeft && root.left == null && root.right == null) {
            sum += root.val;
        }
        plusLeft(root.left, true);
        plusLeft(root.right, false);
    }

    // 513. 找树左下角的值
    // 反向层序遍历？
    public int findBottomLeftValue(TreeNode root) {
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            root = queue.poll();
            if (root.right != null) {
                queue.add(root.right);
            }
            if (root.left != null) {
                queue.add(root.left);
            }
        }
        return root.val;
    }

    // 538. 把二叉搜索树转换为累加树
    // 右根左
    int num = 0;

    public TreeNode convertBST(TreeNode root) {
        if (root != null) {
            // 先右边加
            convertBST(root.right);
            // 再中间加
            num += root.val;
            root.val = num;
            // 最后左边加
            convertBST(root.left);
            // ps 我是智障
        }
        return root;
    }

    // 235. 二叉搜索树的最近公共祖先
    // BST是二叉搜索树（我tm竟然连这个都不知道）
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || p == null || q == null) {
            return null;
        }
        if (root.val < Math.min(p.val, q.val)) {
            return lowestCommonAncestor(root.right, p, q);
        }
        if (root.val > Math.max(p.val, q.val)) {
            return lowestCommonAncestor(root.left, p, q);
        }
        return root;
    }

    // 530. 二叉搜索树的最小绝对差
    // 非负的
    // 中序遍历：左根右
    int last = -1, min = Integer.MAX_VALUE;

    public int getMinimumDifference(TreeNode root) {
        if (root != null) {
            getMinimumDifference(root.left);
            if (last > -1) {
                min = Math.min(min, root.val - last);
            }
            last = root.val;
            getMinimumDifference(root.right);
        }
        return min;
    }

    // 889. 根据前序和后序遍历构造二叉树
    public TreeNode constructFromPrePost(int[] pre, int[] post) {
        if (pre.length == 0) {
            return null;
        }
        Map<Integer, Integer> postMap = new HashMap<>();
        for (int i = 0; i < post.length; i++) {
            // 将值和坐标存进map
            postMap.put(post[i], i);
        }
        return constructFromPrePost(pre, 0, pre.length - 1, postMap, 0);
    }

    TreeNode constructFromPrePost(int[] pre, int preLeft, int preRight, Map<Integer, Integer> postMap, int postLeft) {
        if (preLeft > preRight) {
            return null;
        }
        // 建造根
        int val = pre[preLeft++];
        TreeNode root = new TreeNode(val);
        if (preLeft > preRight) {
            return root;
        }
        int leftIndex = postMap.get(pre[preLeft]);
        int index = leftIndex - postLeft;
        // 建造左子树
        root.left = constructFromPrePost(pre, preLeft, Math.min(preLeft + index, preRight), postMap, postLeft);
        // 建造右子树
        root.right = constructFromPrePost(pre, preLeft + index + 1, preRight, postMap, leftIndex + 1);
        return root;
    }

    // 106. 从中序与后序遍历序列构造二叉树
    // 中序：左根右  后序：左右根
    public TreeNode buildTree2(int[] inorder, int[] postorder) {
        return constructFromInPost(postorder, 0, postorder.length - 1, inorder, 0, inorder.length - 1);
    }

    TreeNode constructFromInPost(int[] postorder, int postLeft, int postRight, int[] inorder, int inLeft, int inRight) {
        if (postLeft > postRight) {
            return null;
        }

        TreeNode root = new TreeNode(postorder[postRight]);
        int inIndex = -1;
        for (int i = inLeft; i <= inRight; i++) {
            if (inorder[i] == postorder[postRight]) {
                inIndex = i;
                break;
            }
        }
        int leftIndex = inIndex - inLeft;
        root.left = constructFromInPost(postorder, postLeft, postLeft + leftIndex - 1, inorder, inLeft, inIndex - 1);
        root.right = constructFromInPost(postorder, postLeft + leftIndex, postRight - 1, inorder, inIndex + 1, inRight);
        return root;
    }

    // 94. 二叉树的中序遍历
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        inorderTravel(root, res);
        return res;
    }

    void inorderTravel(TreeNode root, List<Integer> res) {
        if (root == null) {
            return;
        }
        inorderTravel(root.left, res);
        res.add(root.val);
        inorderTravel(root.right, res);
    }

    // 非递归写法
    public List<Integer> inorderTraversal2(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Stack<TreeNode> stack = new Stack<>();
        while (!stack.empty() || root != null) {
            if (root != null) {
                stack.push(root);
                root = root.left;
            } else {
                // 将左节点入队
                root = stack.pop();
                res.add(root.val);
                root = root.right;
            }
        }

        return res;
    }

    // 145. 二叉树的后序遍历
    // 左右根
    // 用非递归方法写
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Stack<TreeNode> stack = new Stack<>();
        stack.add(root);
        while (!stack.empty()) {
            TreeNode node = stack.pop();
            if (node.left != null) {
                stack.push(node.left);
            }
            if (node.right != null) {
                stack.push(node.right);
            }
            res.add(0, node.val);
        }

        return res;
    }

    // 236. 二叉树的最近公共祖先
    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {
            return null;
        }
        if (root == p || root == q) {
            return root;
        }
        TreeNode left = lowestCommonAncestor2(root.left, p, q);
        TreeNode right = lowestCommonAncestor2(root.right, p, q);
        if (left != null && right != null) {
            return root;
        } else if (left != null) {
            return left;
        } else if (right != null) {
            return right;
        }
        return null;
    }

    // 109. 有序链表转换二叉搜索树
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) {
            return null;
        }
        // 若此时链表只剩下一个数，则无须在进行以下的链表分割步骤
        if (head.next == null) {
            return new TreeNode(head.val);
        }
        // 利用快慢指针找到链表的中点
        ListNode pre = null;
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            pre = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        // 构建中间节点，并将左右子树递归处理
        TreeNode root = new TreeNode(slow.val);
        // 截断链表
        // pre不可能为null，所以不必担心
        pre.next = null;
        root.left = sortedListToBST(head);
        root.right = sortedListToBST(slow.next);

        return root;
    }

    // 897. 递增顺序搜索树
    // 二叉搜索树，中序遍历（左根右）
    TreeNode tmp;

    public TreeNode increasingBST(TreeNode root) {
        if (root == null) {
            return null;
        }
        TreeNode res = new TreeNode();
        tmp = res;
        increasing(root);

        return res.right;
    }

    void increasing(TreeNode node) {
        if (node == null) {
            return;
        }
        increasing(node.left);
        tmp.right = node;
        node.left = null;
        tmp = node;
        increasing(node.right);
    }

    // 653. 两数之和 IV - 输入 BST
    public boolean findTarget(TreeNode root, int k) {
        if (root == null) {
            return false;
        }
        Set<Integer> set = new HashSet<>();
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (set.contains(k - node.val)) {
                return true;
            }
            set.add(node.val);
            if (node.left != null) {
                queue.add(node.left);
            }
            if (node.right != null) {
                queue.add(node.right);
            }
        }

        return false;
    }

    Set<Integer> valSet = new HashSet<>();

    // 不知道为啥，这个递归的层序遍历比上面那种用队列的层序遍历快多了
    public boolean findTarget2(TreeNode root, int k) {
        if (root == null) {
            return false;
        }
        if (valSet.contains(k - root.val)) {
            return true;
        }
        valSet.add(root.val);
        return findTarget2(root.left, k) || findTarget2(root.right, k);
    }

    // 450. 删除二叉搜索树中的节点
    public TreeNode deleteNode(TreeNode root, int key) {
        if (root == null) {
            return null;
        }
        if (root.val < key) {
            root.right = deleteNode(root.right, key);
        } else if (root.val > key) {
            root.left = deleteNode(root.left, key);
        } else {
            if (root.left == null || root.right == null) {
                // 若命中点只有少于等于一个子树
                root = root.left == null ? root.right : root.left;
            } else {
                // 命中点左右子树都存在
                TreeNode tmp = root.right;
                while (tmp.left != null) {
                    tmp = tmp.left;
                }
                root.val = tmp.val;
                root.right = deleteNode(root.right, tmp.val);
            }
        }
        return root;
    }

    public static void main(String[] args) {
        Tree tree = new Tree();

        TreeNode node1 = new TreeNode(5);
        TreeNode node2 = new TreeNode(3);
        TreeNode node3 = new TreeNode(6);
        TreeNode node4 = new TreeNode(2);
        TreeNode node5 = new TreeNode(4);
        TreeNode node6 = new TreeNode(7);
        node1.left = node2;
        node1.right = node3;
        node2.left = node4;
        node2.right = node5;
        node3.right = node6;

        System.out.println(tree.deleteNode(node1, 3));
    }
}

// 208. 实现 Trie (前缀树)
class Trie {
    TrieNode root; // 根节点

    /**
     * Initialize your data structure here.
     */
    public Trie() {
        root = new TrieNode();
    }

    /**
     * Inserts a word into the trie.
     */
    public void insert(String word) {
        TrieNode node = root;
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            node = node.addChild(ch);
        }
        node.canEnd = true;
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        TrieNode node = root;
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            TrieNode child = node.containsChild(ch);
            if (child == null) {
                return false;
            }
            node = child;
        }
        return node.canEnd;
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        TrieNode node = root;
        for (int i = 0; i < prefix.length(); i++) {
            char ch = prefix.charAt(i);
            TrieNode child = node.containsChild(ch);
            if (child == null) {
                return false;
            }
            node = child;
        }
        return true;
    }
}

class TrieNode {
    char val; // 节点值
    boolean canEnd = false;
    TrieNode[] childNodes = new TrieNode[26]; // 子节点，26个字母

    TrieNode containsChild(char ch) {
        return childNodes[ch - '0' - 49];
    }

    TrieNode addChild(char ch) {
        TrieNode child;
        child = containsChild(ch);
        if (child == null) {
            child = new TrieNode();
            child.val = ch;
            childNodes[ch - '0' - 49] = child;
        }
        return child;
    }
}
