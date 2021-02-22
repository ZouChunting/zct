package leetcode;

import java.util.Arrays;

public class BinarySearch {
    // 69. x 的平方根
    // 牛顿迭代法，容后再看
    public int mySqrt(int x) {
        if (x == 1) {
            return x;
        }
        // 这里只能用0起始，否则会溢出
        int left = 0;
        int right = x;
        while (right - left > 1) {
            int mid = (left + right) / 2;
            if (x / mid < mid) {
                right = mid;
            } else if (x / mid > mid) {
                left = mid;
            } else {
                return mid;
            }
        }
        return left;
    }

    // 34. 在排序数组中查找元素的第一个和最后一个位置
    // 升序
    public int[] searchRange(int[] nums, int target) {
        int[] res = new int[]{-1, -1};
        if (nums.length == 0) {
            return res;
        }
        int left = 0;
        int right = nums.length - 1;
        // 寻找左边界
        while (left < right) {
            int m = (left + right) / 2;
            if (nums[m] < target) {
                // 若当前值小于目标值，向右逐步逼近
                left = m + 1;
            } else {
                // 若当前值大于等于目标值，将右坐标置于此处
                right = m;
            }
        }
        if (nums[left] != target) {
            return res;
        }
        res[0] = left;

        // 寻找右边界
        // {1} 1：此处用nums.length而不用nums.length - 1
        right = nums.length;
        while (left < right) {
            int m = (left + right) / 2;
            if (nums[m] > target) {
                // 若当前值大于目标值，将有坐标置于此处
                right = m;
            } else {
                // 若当前值小于等于目标值，向右逐步逼近
                left = m + 1;
            }
        }
        res[1] = left - 1;

        return res;
    }

    // 81. 搜索旋转排序数组 II
    // 升序
    public boolean search(int[] nums, int target) {
        if (nums.length == 0) {
            return false;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[left] == target || nums[right] == target || nums[mid] == target) {
                return true;
            }
            if (nums[mid] > target) {
                // 当前值大于目标值
                if (nums[left] < nums[mid]) {
                    // 左区间有序
                    if (nums[left] < target) {
                        // 左端小于目标值
                        right = mid;  // 探索左区间
                    } else {
                        left = mid;  // 探索右区间
                    }
                } else if (nums[mid] < nums[right] || (nums[left] > nums[mid] && nums[mid] == nums[right])) {
                    // 右区间有序
                    right = mid;  // 探索左区间
                }
            } else if (nums[mid] < target) {
                // 当前值小于目标值
                if (nums[left] < nums[mid]) {
                    // 左区间有序
                    left = mid; // 探索右区间
                } else if (nums[mid] < nums[right] || (nums[left] > nums[mid] && nums[mid] == nums[right])) {
                    // 右区间有序
                    if (target < nums[right]) {
                        // 右端大于目标值
                        left = mid; // 探索右区间
                    } else {
                        right = mid;  // 探索左区间
                    }
                }
            }
            left++;
            right--;
        }

        return false;
    }

    // 153. 寻找旋转排序数组中的最小值
    // 154. 寻找旋转排序数组中的最小值II
    public int findMin(int[] nums) {
        int res = nums[0];
        // 排除数组过短情况和无反转情况
        if (nums.length == 1 || nums[0] < nums[nums.length - 1]) {
            return res;
        }
        int left = 0;
        int right = nums.length - 1;
        if (nums[left] > nums[left + 1]) {
            return nums[left + 1];
        }
        if (nums[right] < nums[right - 1]) {
            return nums[right];
        }
        // 找出翻转点
        while (left <= right) {
            int mid = (left + right) / 2;
            if (mid + 1 <= nums.length - 1 && nums[mid + 1] < nums[mid]) {
                return nums[mid + 1];
            }
            if (mid - 1 >= 0 && nums[mid - 1] > nums[mid]) {
                return nums[mid];
            }

            if (nums[left] < nums[mid]) {
                // 左区间有序
                left = mid + 1; // 探索右区间
            } else if (nums[mid] < nums[right] || (nums[left] > nums[mid] && nums[mid] == nums[right])) {
                // 右区间有序
                right = mid - 1; // 探索左区间
            } else {
                // 无法判断有序
                if (left + 1 <= nums.length - 1 && nums[left] > nums[left + 1]) {
                    return nums[left + 1];
                }
                left++;
            }
        }
        return res;
    }

    // 540. 有序数组中的单一元素
    // 升序
    public int singleNonDuplicate(int[] nums) {
        // 数组长度必为奇数是重要条件，是while和return重要依据
        if (nums.length == 1) {
            return nums[0];
        }
        int left = 0, right = nums.length - 1, mid = 0;
        while (left < right) {
            mid = (left + right) / 2;
            // 0 1 2 3
            if (nums[mid] == nums[mid - 1]) {
                if ((mid - left) % 2 == 0) {
                    // 向左区间寻找
                    right = mid - 2;
                } else {
                    // 向右区间寻找
                    left = mid + 1;
                }
            } else if (nums[mid] == nums[mid + 1]) {
                // 0 1 2 3 4
                if ((mid - left) % 2 == 0) {
                    // 向右区间寻找
                    left = mid + 2;
                } else {
                    // 向左区间寻找
                    right = mid - 1;
                }
            } else {
                return nums[mid];
            }
        }
        // left和right都可，因为left和right最后必相等
        return nums[left];
    }

    // 4. 寻找两个正序数组的中位数
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int len1 = nums1.length;
        int len2 = nums2.length;
        int len = len1 + len2;
        double median;
        if (len % 2 == 0) {
            // 中位数是两数均值
            median = (getKthElement(nums1, nums2, len / 2) + getKthElement(nums1, nums2, len / 2 + 1)) / 2.0;
        } else {
            // 中位数只有一个
            // 0 1 2 3 4 长度为5，寻找第3小的数
            median = getKthElement(nums1, nums2, len / 2 + 1);
        }
        return median;
    }

    public int getKthElement(int[] nums1, int[] nums2, int k) {
        // 寻找两个数组中第k个小的数
        int len1 = nums1.length;
        int len2 = nums2.length;
        int index1 = 0;
        int index2 = 0;
        while (true) {
            // nums1数组已为空
            if (index1 == len1) {
                return nums2[index2 + k - 1];
            }
            // nums2数组已为空
            if (index2 == len2) {
                return nums1[index1 + k - 1];
            }
            // 在得出结果之前做数组判空十分必要
            if (k == 1) {
                return Math.min(nums1[index1], nums2[index2]);
            }
            // 两个数组先各自找目标坐标的一半
            int mid = k / 2;
            int newIndex1 = Math.min(index1 + mid, len1) - 1;
            int newIndex2 = Math.min(index2 + mid, len2) - 1;
            if (nums1[newIndex1] <= nums2[newIndex2]) {
                k = k - (newIndex1 - index1 + 1);
                index1 = newIndex1 + 1;
            } else {
                k = k - (newIndex2 - index2 + 1);
                index2 = newIndex2 + 1;
            }
        }
    }

    public static void main(String[] args) {
        BinarySearch binarySearch = new BinarySearch();

        int x = 2147483647;
        // System.out.println(binarySearch.mySqrt(x));

        // System.out.println(46340 * 46340);
        // System.out.println(1 + 2147483647);

        // int[] nums = {1, 1, 1, 1, 1, 1, 1, 1, 1, 13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        // int target = 13;
        // System.out.println(Arrays.toString(binarySearch.searchRange(nums, target)));

        // System.out.println(binarySearch.search(nums, target));

        // int[] nums = {150, 151, 152, 156, 158, 159, 160, 161, 162, 167, 169, 170, 171, 177, 180, 183, 184, 186, 189, 191, 197, 200, 203, 205, 210, 215, 216, 219, 221, 222, 233, 236, 237, 238, 239, 246, 247, 250, 254, 257, 260, 261, 262, 269, 275, 279, 283, 284, 286, 287, 288, 289, 290, 294, 295, 298, 1, 5, 6, 9, 10, 13, 15, 16, 20, 25, 27, 28, 34, 37, 41, 42, 43, 46, 48, 51, 53, 54, 59, 61, 65, 67, 72, 76, 78, 79, 81, 83, 85, 91, 92, 94, 95, 102, 103, 105, 106, 111, 113, 118, 120, 122, 123, 126, 141, 148};
        // System.out.println(binarySearch.findMin(nums));

        // int[] nums = {3, 3, 1, 3, 3, 3, 3};
        // System.out.println(binarySearch.findMin(nums));

        int[] nums = {1, 1, 2, 3, 3, 4, 4, 8, 8};
        // System.out.println(binarySearch.singleNonDuplicate(nums));

        int[] nums1 = {3};
        int[] nums2 = {-2,-1};
        System.out.println(binarySearch.findMedianSortedArrays(nums1, nums2));
    }
}
