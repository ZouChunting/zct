package leetcode;

public class Chain {
    // 链表
    // 206. 反转链表
    // 递归写法
    public ListNode reverseList(ListNode head) {
        if (head.next == null) {
            return head;
        }
        ListNode tmp = head;
        head = reverseList(head.next);
        head.next = tmp;
        return tmp;
    }

    public ListNode reverseList(ListNode head, ListNode tmp) {
        if (head == null) {
            return tmp;
        }
        ListNode next = head.next;
        head.next = tmp;
        return reverseList(next, head);
    }

    // 非递归写法
    // 1->2->3 3->2->1
    public ListNode reverseList2(ListNode head) {
        ListNode tmp = null, p;
        while (head != null) {
            p = head; // 获取当前节点
            head = head.next; // 记录下一节点
            p.next = tmp; // 翻转指向
            tmp = p; // 保存翻转节点
        }
        return tmp;
    }

    // 21. 合并两个有序链表
    // 递归
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null && l2 == null) {
            return null;
        }
        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }
        ListNode res;
        if (l1.val <= l2.val) {
            res = l1;
            l1 = l1.next;
        } else {
            res = l2;
            l2 = l2.next;
        }

        res.next = mergeTwoLists(l1, l2);
        return res;
    }

    // 非递归
    // 注意指针
    public ListNode mergeTwoLists2(ListNode l1, ListNode l2) {
        ListNode root = new ListNode();
        ListNode tmp = root;
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                tmp.next = l1;
                l1 = l1.next;
            } else {
                tmp.next = l2;
                l2 = l2.next;
            }
            tmp = tmp.next;
        }
        while (l1 != null) {
            tmp.next = l1;
            l1 = l1.next;
            tmp = tmp.next;
        }
        while (l2 != null) {
            tmp.next = l2;
            l2 = l2.next;
            tmp = tmp.next;
        }
        return root.next;
    }

    // 24. 两两交换链表中的节点
    public ListNode swapPairs(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode next = head.next;
        // 交换两节点
        head.next = next.next;
        next.next = head;
        head.next = swapPairs(head.next);

        return next;
    }

    // 160. 相交链表
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode nodeA = headA, nodeB = headB;
        while (nodeA != nodeB) {
            nodeA = nodeA == null ? headB : nodeA.next;
            nodeB = nodeB == null ? headA : nodeB.next;
        }
        return nodeA;
    }

    // 234. 回文链表
    // 1 2 3 4 5 6 7 8 9
    public boolean isPalindrome(ListNode head) {
        // 通过快慢指针找到中点，再把前半部分的指向颠倒一下
        ListNode slow = head;
        ListNode fast = head;
        ListNode tmp = null;
        ListNode h = slow;
        boolean flag = false;
        while (fast != null) {
            if (fast.next == null) {
                // 节点个数为奇数
                flag = true;
                break;
            }
            fast = fast.next.next;
            h = slow;
            slow = slow.next;
            h.next = tmp;
            tmp = h;
        }
        if (flag) {
            slow = slow.next;
        }
        while (slow != null) {
            if (slow.val != h.val) {
                return false;
            }
            slow = slow.next;
            h = h.next;
        }
        return true;
    }

    // 83. 删除排序链表中的重复元素
    // 2 2 2 2 3
    // 1 2 3 4 5
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode pre = head;
        ListNode cur = head.next;
        while (cur != null) {
            if (cur.val == pre.val) {
                cur = cur.next;
                pre.next = cur;
            } else {
                cur = cur.next;
                pre = pre.next;
            }
        }
        return head;
    }

    // 328. 奇偶链表
    public ListNode oddEvenList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode odd = head; // 奇数
        ListNode even = head.next; // 偶数
        while (odd != null) {
            odd = odd.next;
            even = even.next;
        }
        return head;
    }

    // 19. 删除链表的倒数第 N 个结点
    // head = [1,2,3,4,5], n = 2
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode tmp = head;
        while (n > 0) {
            tmp = tmp.next;
            n--;
        }
        ListNode node = null;
        while (tmp != null) {
            node = node.next;
            tmp = tmp.next;
        }
        node.next = node.next.next;

        return head;
    }

    public static void main(String[] args) {
        Chain chain = new Chain();

        ListNode node1 = new ListNode(1);
        ListNode node2 = new ListNode(2);
        ListNode node3 = new ListNode(1);

        // node1.next = node2;
        // node2.next = node3;

        System.out.println(chain.removeNthFromEnd(node1, 1));
    }
}