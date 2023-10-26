//FormAI DATASET v0.1 Category: Clustering Algorithm Implementation ; Style: curious
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 100 // maximum number of points
#define K 2 // number of clustering dimensions
#define MAX_ITER 100 // maximum number of iterations
#define EPSILON 0.0001 // convergence criterion

typedef struct point {
    double coords[K];
    int cluster;
} Point;

Point points[N];
int num_points;
int num_clusters;

// function to initialize points
void init_points() {
    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < K; j++) {
            points[i].coords[j] = rand() % 100;
        }
        points[i].cluster = -1;
    }
}

// function to compute distance between two points
double compute_distance(Point p1, Point p2) {
    double distance = 0;
    for (int i = 0; i < K; i++) {
        distance += pow((p1.coords[i] - p2.coords[i]), 2);
    }
    return sqrt(distance);
}

// function to assign points to nearest cluster
void assign_clusters() {
    for (int i = 0; i < num_points; i++) {
        double min_distance = INFINITY;
        for (int j = 0; j < num_clusters; j++) {
            double distance = compute_distance(points[i], points[j]);
            if (distance < min_distance) {
                min_distance = distance;
                points[i].cluster = j;
            }
        }
    }
}

// function to update cluster means
void update_means() {
    double clusters[num_clusters][K];
    int counts[num_clusters];
    for (int i = 0; i < num_clusters; i++) {
        for (int j = 0; j < K; j++) {
            clusters[i][j] = 0;
        }
        counts[i] = 0;
    }
    for (int i = 0; i < num_points; i++) {
        int cluster = points[i].cluster;
        for (int j = 0; j < K; j++) {
            clusters[cluster][j] += points[i].coords[j];
        }
        counts[cluster]++;
    }
    for (int i = 0; i < num_clusters; i++) {
        for (int j = 0; j < K; j++) {
            if (counts[i] > 0) {
                clusters[i][j] /= counts[i];
            }
        }
    }
    for (int i = 0; i < num_points; i++) {
        int cluster = points[i].cluster;
        for (int j = 0; j < K; j++) {
            points[i].coords[j] = clusters[cluster][j];
        }
    }
}

// function to check for convergence
int check_convergence(Point old_points[]) {
    for (int i = 0; i < num_points; i++) {
        if (compute_distance(points[i], old_points[i]) > EPSILON) {
            return 0;
        }
    }
    return 1;
}

int main() {
    printf("Welcome to the Curious C Clustering Algorithm!\n");
    printf("Please enter the number of points to cluster: ");
    scanf("%d", &num_points);
    printf("Please enter the number of clusters to use: ");
    scanf("%d", &num_clusters);
    init_points();
    Point old_points[num_points];
    for (int i = 0; i < num_points; i++) {
        old_points[i] = points[i];
    }
    int iter = 0;
    while (iter < MAX_ITER) {
        assign_clusters();
        update_means();
        if (check_convergence(old_points) == 1) {
            printf("Convergence achieved after %d iterations!\n", iter+1);
            break;
        } else {
            for (int i = 0; i < num_points; i++) {
                old_points[i] = points[i];
            }
        }
        iter++;
    }
    for (int i = 0; i < num_points; i++) {
        printf("Point %d: [", i+1);
        for (int j = 0; j < K; j++) {
            printf("%.2f", points[i].coords[j]);
            if (j < K-1) {
                printf(", ");
            }
        }
        printf("] (Cluster %d)\n", points[i].cluster+1);
    }
    printf("Thank you for trying out the Curious C Clustering Algorithm! Goodbye :)\n");
    return 0;
} 
