package leetcode;

import java.util.Arrays;

public class Solution1 {
    // 4. 寻找两个正序数组的中位数
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int len1 = nums1.length;
        int len2 = nums2.length;
        int len = len1 + len2;
        double median;
        if(len % 2 == 1){
            // 总长度为奇数
            int index = len / 2 + 1;
            median = getKthElement(nums1, nums2, index);
        }else {
            // 总长度为偶数
            int index1 = len / 2, index2 = len / 2 + 1;
            median = (getKthElement(nums1, nums2, index1) + getKthElement(nums1, nums2, index2)) / 2.0;
        }
        return median;
    }

    public int getKthElement(int[] nums1, int[] nums2, int k){
        int len1 = nums1.length;
        int len2 = nums2.length;
        int index1 = 0, index2 = 0;

        while(true){
            if(index1 == len1){
                return nums2[index2 + k - 1];
            }

            if(index2 == len2){
                return nums1[index1 + k - 1];
            }

            if(k == 1){
                return Math.min(nums1[index1], nums2[index2]);
            }

            int half = k / 2;
            int newIndex1 = Math.min(index1 + half, len1) - 1;
            int newIndex2 = Math.min(index2 + half, len2) - 1;
            if(nums1[newIndex1] <= nums2[newIndex2]){
                k = k - (newIndex1 - index1 + 1);
                index1 = newIndex1 + 1;
            }else {
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
        for(int i=1; i<len; i++){
            int cur = nums[i];
            if(pre <= 0){
                pre = cur;
            }else {
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
        while(true){
            if(index1 < 0 || index2 < 0){
                break;
            }
            if(nums2[index2] > nums1[index1]){
                nums1[p] = nums2[index2];
                index2 --;
            }else {
                nums1[p] = nums1[index1];
                index1 --;
            }
            p --;
        }
    }

    public static void main(String[] args){
        Solution1 s1 = new Solution1();

        /*
        int[] nums1 = {1,2};
        int[] nums2 = {3,4};
        double res = s1.findMedianSortedArrays(nums1, nums2);
        System.out.println(res);

        int[] nums = {-2,1,-3,4,-1,2,1,-5,4};
        System.out.println(s1.maxSubArray1(nums));
        */

        int[] nums1 = {0};
        int[] nums2 = {1};
        int m = 0, n = 1;
        s1.merge(nums1, m, nums2, n);
        System.out.println(Arrays.toString(nums1));
    }
}
