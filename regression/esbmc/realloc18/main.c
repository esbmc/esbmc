#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void check_ptr(void *ptr, const char *msg) {
    if (!ptr) {
        perror(msg);
        exit(EXIT_FAILURE);
    }
}

/*
 * Test 1: Dynamic vector growth using realloc in a loop
 * Mimics push_back in a C++ vector.
 */
void test_vector_growth() {
    printf("Test 1: vector growth\n");
    int capacity = 1;
    int size = 0;
    int *vec = malloc(capacity * sizeof(int));
    check_ptr(vec, "malloc");

    for (int i = 0; i < 20; i++) {
        if (size == capacity) {
            capacity *= 2;
            int *tmp = realloc(vec, capacity * sizeof(int));
            check_ptr(tmp, "realloc");
            vec = tmp;
        }
        vec[size++] = i;
    }

    for (int i = 0; i < size; i++)
        printf("%d ", vec[i]);
    printf("\n");
    free(vec);
}

/*
 * Test 2: Shrinking vector (simulating pop_back with shrink_to_fit)
 */
void test_vector_shrink() {
    printf("Test 2: vector shrink\n");
    int capacity = 16;
    int size = capacity;
    int *vec = malloc(capacity * sizeof(int));
    check_ptr(vec, "malloc");

    for (int i = 0; i < size; i++) vec[i] = i;

    // Shrink step by step
    for (int new_cap = capacity; new_cap > 0; new_cap /= 2) {
        int *tmp = realloc(vec, new_cap * sizeof(int));
        check_ptr(tmp, "realloc shrink");
        vec = tmp;
        size = new_cap;
        printf("New size=%d, last element=%d\n", size, vec[size - 1]);
    }

    free(vec);
}

/*
 * Test 3: Alternating shrink and expand
 * Stress data preservation when realloc moves memory around.
 */
void test_alternating_resize() {
    printf("Test 3: alternating shrink/expand\n");
    int n = 8;
    int *arr = malloc(n * sizeof(int));
    check_ptr(arr, "malloc");

    for (int i = 0; i < n; i++) arr[i] = i;

    for (int round = 0; round < 5; round++) {
        // shrink
        int shrink_n = n / 2;
        int *tmp = realloc(arr, shrink_n * sizeof(int));
        check_ptr(tmp, "realloc shrink");
        arr = tmp;
        n = shrink_n;
        printf("Shrink round %d: size=%d first=%d last=%d\n", round, n, arr[0], arr[n - 1]);

        // expand
        int expand_n = n * 3;
        tmp = realloc(arr, expand_n * sizeof(int));
        check_ptr(tmp, "realloc expand");
        arr = tmp;
        // fill new area
        for (int i = n; i < expand_n; i++) arr[i] = i;
        n = expand_n;
        printf("Expand round %d: size=%d first=%d last=%d\n", round, n, arr[0], arr[n - 1]);
    }

    free(arr);
}

/*
 * Test 4: Resize to zero, then grow again
 */
void test_resize_zero_grow() {
    printf("Test 4: zero then grow\n");
    int *arr = malloc(5 * sizeof(int));
    check_ptr(arr, "malloc");
    for (int i = 0; i < 5; i++) arr[i] = i;

    // Shrink to zero
    int *tmp = realloc(arr, 0);
    arr = tmp; // likely NULL, memory freed
    printf("Shrink to zero done\n");

    // Grow again from NULL
    arr = realloc(NULL, 10 * sizeof(int));
    check_ptr(arr, "realloc grow from NULL");
    for (int i = 0; i < 10; i++) arr[i] = i * 2;
    printf("Regrown: first=%d last=%d\n", arr[0], arr[9]);

    free(arr);
}

int main(void) {
    test_vector_growth();
    test_vector_shrink();
    test_alternating_resize();
    test_resize_zero_grow();
    return 0;
}

