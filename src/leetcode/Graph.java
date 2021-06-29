package leetcode;

import java.util.*;

public class Graph {
    // 图
    // 785. 判断二分图
    public boolean isBipartite(int[][] graph) {
        int n = graph.length;  // 点的个数
        int[] color = new int[n]; // 点的着色，0代表未着色，有0、1两种颜色
        Queue<Integer> queue = new ArrayDeque<>();

        for (int i = 0; i < n; i++) {
            if (color[i] == 0) {
                // 该点还未染色
                color[i] = 1;
                queue.add(i);
            }
            while (!queue.isEmpty()) {
                int node = queue.poll();
                // 遍历与node相连的点
                for (int k : graph[node]) {
                    if (color[k] == 0) {
                        color[k] = color[node] == 1 ? 2 : 1;
                        queue.add(k);
                    } else {
                        if (color[k] == color[node]) {
                            return false;
                        }
                    }
                }
            }
        }

        return true;
    }

    // 210. 课程表 II
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] res = new int[numCourses];
        int index = 0;
        int[][] graph = new int[numCourses][numCourses];  // 邻接矩阵
        for (int[] prerequisite : prerequisites) {
            int end = prerequisite[0];  // 边的终点
            int start = prerequisite[1];  // 边的起点
            graph[start][end] = 1;
            graph[end][end] += 1;  // 存储该点的入度
        }
        Queue<Integer> queue = new ArrayDeque<>();
        for (int i = 0; i < numCourses; i++) {
            if (graph[i][i] == 0) {
                // 将入度为0的点加入队列
                queue.add(i);
            }
        }
        while (!queue.isEmpty()) {
            int node = queue.poll();
            res[index++] = node;
            for (int i = 0; i < numCourses; i++) {
                if (graph[node][i] > 0) {
                    graph[node][i] = 0;
                    graph[i][i]--;
                    if (graph[i][i] == 0) {
                        queue.add(i);
                    }
                }
            }
        }

        return index < numCourses ? new int[0] : res;
    }

    // 882. 细分图中的可到达结点
    public int reachableNodes(int[][] edges, int maxMoves, int n) {
        return 0;
    }

    public static void main(String[] args) {
        Graph graph = new Graph();

        int[][] graphArr = {
                {0, 1},
                {1, 0}
        };
        System.out.println(Arrays.toString(graph.findOrder(2, graphArr)));
    }
}
