package leetcode;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;

public class Sort {
    public void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

    // 快速排序
    public void quickSort(int[] nums, int left, int right) {
        if (left < right) {
            int key = nums[left];
            int i = left;
            int j = right;
            while (i < j) {
                while (i < j && nums[j] >= key) {
                    j--;
                }
                if (i < j) {
                    nums[i] = nums[j];
                    i++;
                }
                while (i < j && nums[i] <= key) {
                    i++;
                }
                if (i < j) {
                    nums[j] = nums[i];
                    j--;
                }
            }
            nums[i] = key;
            quickSort(nums, left, i - 1);
            quickSort(nums, i + 1, right);
        }
    }

    // 归并排序
    // 归并排序可细看，深入了解递归（可以[0,8]区间向下划分区间的二叉树形式理解）
    public void mergeSort(int[] nums, int left, int right, int[] tmp) {
        if (left < right) {
            int mid = (left + right) / 2;
            mergeSort(nums, left, mid, tmp);
            mergeSort(nums, mid + 1, right, tmp);
            int i = left, j = mid + 1, t = 0;
            while (i <= mid && j <= right) {
                if (nums[i] <= nums[j]) {
                    tmp[t++] = nums[i++];
                } else {
                    tmp[t++] = nums[j++];
                }
            }
            while (i <= mid) {
                tmp[t++] = nums[i++];
            }
            while (j <= right) {
                tmp[t++] = nums[j++];
            }
            t = 0;
            while (left <= right) {
                nums[left++] = tmp[t++];
            }
        }
    }

    // 插入排序
    public void insertionSort(int[] nums) {
        int i, j, tmp;
        for (i = 1; i < nums.length; i++) {
            tmp = nums[i];
            for (j = i - 1; j >= 0 && nums[j] > tmp; j--) {
                nums[j + 1] = nums[j];
            }
            nums[j + 1] = tmp;
        }
    }

    // 冒泡排序
    // 简单理解就是，每次都把最大的放后面
    public void bubbleSort(int[] nums) {
        int i, j, tmp;
        for (i = nums.length - 1; i > 0; i--) {
            for (j = 0; j < i; j++) {
                if (nums[j] > nums[j + 1]) {
                    tmp = nums[j + 1];
                    nums[j + 1] = nums[j];
                    nums[j] = tmp;
                }
            }
        }
    }

    // 选择排序
    public void selectionSort(int[] nums) {
        int i, j, index, tmp;
        for (i = 0; i < nums.length - 1; i++) {
            index = i;
            for (j = i + 1; j < nums.length; j++) {
                if (nums[j] < nums[index]) {
                    index = j;
                }
            }
            if (index != i) {
                tmp = nums[index];
                nums[index] = nums[i];
                nums[i] = tmp;
            }
        }
    }

    // 215. 数组中的第K个最大元素
    // 快排加二分
    public int findKthLargest(int[] nums, int k) {
        int left = 0, right = nums.length - 1;
        int target = nums.length - k;
        while (true) {
            int index = quickSelection(nums, left, right);
            if (index == target) {
                return nums[index];
            } else if (index < target) {
                left = index + 1;
            } else {
                right = index - 1;
            }
        }
    }

    public int quickSelection(int[] nums, int left, int right) {
        int key = nums[left];
        int i = left;
        int j = right;
        while (i < j) {
            while (i < j && nums[j] > key) {
                j--;
            }
            while (i < j && nums[i] < key) {
                i++;
            }
            if (i < j) {
                swap(nums, i, j);
            }
        }
        swap(nums, i, left);
        return i;
    }

    // 347. 前 K 个高频元素
    public int[] topKFrequent(int[] nums, int k) {
        // 统计数组
        Map<Integer, Integer> count = new HashMap<>();
        for (int num : nums) {
            count.put(num, count.getOrDefault(num, 0) + 1);
        }
        // 大顶堆
        PriorityQueue<Integer> pq = new PriorityQueue<>((o1, o2) -> count.get(o2) - count.get(o1));
        pq.addAll(count.keySet());
        int[] res = new int[k];
        for (int i = 0; i < k; i++) {
            res[i] = pq.remove();
        }
        return res;
    }

    // 451. 根据字符出现频率排序
    public String frequencySort(String s) {
        char[] cs = s.toCharArray();
        Map<Character, Integer> count = new HashMap<>();
        for (char c : cs) {
            count.put(c, count.getOrDefault(c, 0) + 1);
        }
        PriorityQueue<Map.Entry<Character, Integer>> pq = new PriorityQueue<>((o1, o2) -> o2.getValue() - o1.getValue());
        pq.addAll(count.entrySet());
        StringBuilder sb = new StringBuilder();
        while (!pq.isEmpty()) {
            Map.Entry<Character, Integer> entry = pq.remove();
            char key = entry.getKey();
            int value = entry.getValue();
            while (value > 0) {
                sb.append(key);
                value--;
            }
        }
        return sb.toString();
    }

    // 75. 颜色分类
    public void sortColors(int[] nums) {
        int i = 0, zero = 0, two = nums.length - 1;
        while (i <= two) {
            // 可注意到，i zero two三者变化过程中，i和two靠近，i始终大于或等于zero
            // 保证i始终在zero前面非常重要
            if (nums[i] == 0) {
                // 0
                swap(nums, i, zero);
                zero++;
                i++;
            } else if (nums[i] == 1) {
                i++;
            } else {
                swap(nums, i, two);
                two--;
            }
        }
    }

    public void sortColors2(int[] nums) {
        int zero = 0, two = nums.length - 1;
        int i = two;
        while (i >= zero) {
            if (nums[i] == 2) {
                swap(nums, i, two);
                two--;
                i--;
            } else if (nums[i] == 1) {
                i--;
            } else {
                swap(nums, i, zero);
                zero++;
            }
        }
    }

    public static void main(String[] args) {
        Sort sort = new Sort();

        // int[] nums = {5, 9, 1, 9, 5, 3, 7, 6, 1};
        // sort.quickSort(nums, 0, nums.length - 1);
        // int[] tmp = new int[nums.length];
        // sort.mergeSort(nums, 0, nums.length - 1, tmp);
        // sort.insertionSort(nums);
        // sort.bubbleSort(nums);
        // sort.selectionSort(nums);
        // System.out.println(Arrays.toString(nums));

        // int[] nums = {1};
        // int k = 1;
        // System.out.println(sort.findKthLargest(nums, k));

        // System.out.println(Arrays.toString(sort.topKFrequent(nums, k)));

        // String s = "Aabb";
        // System.out.println(sort.frequencySort(s));

        int[] nums = {2, 0, 1};
        sort.sortColors2(nums);
        System.out.println(Arrays.toString(nums));
    }
}
