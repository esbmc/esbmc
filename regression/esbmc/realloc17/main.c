#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper to check allocation results
void check_ptr(void *ptr, const char *msg) {
    if (!ptr) {
        perror(msg);
        exit(EXIT_FAILURE);
    }
}

/*
 * Test 1: realloc(NULL, size) behaves like malloc(size).
 */
void test_realloc_null_malloc() {
    printf("Test 1: realloc(NULL, 10)\n");
    int *p = realloc(NULL, 10 * sizeof(int));
    check_ptr(p, "realloc(NULL, 10)");
    free(p);
}

/*
 * Test 2: realloc(ptr, 0) should free memory and return NULL (implementation-defined).
 */
void test_realloc_zero() {
    printf("Test 2: realloc(ptr, 0)\n");
    int *p = malloc(10 * sizeof(int));
    check_ptr(p, "malloc");
    int *q = realloc(p, 0);
    // q may be NULL; p is freed regardless
    if (q == NULL)
        printf("Returned NULL as expected\n");
    else
        free(q);
}

/*
 * Test 3: Shrinking allocated memory.
 */
void test_realloc_shrink() {
    printf("Test 3: realloc shrink\n");
    int *p = malloc(10 * sizeof(int));
    check_ptr(p, "malloc");
    for (int i = 0; i < 10; i++) p[i] = i;
    int *q = realloc(p, 5 * sizeof(int));
    check_ptr(q, "realloc shrink");
    for (int i = 0; i < 5; i++)
        printf("%d ", q[i]);
    printf("\n");
    free(q);
}

/*
 * Test 4: Expanding allocated memory.
 */
void test_realloc_expand() {
    printf("Test 4: realloc expand\n");
    int *p = malloc(5 * sizeof(int));
    check_ptr(p, "malloc");
    for (int i = 0; i < 5; i++) p[i] = i;
    int *q = realloc(p, 10 * sizeof(int));
    check_ptr(q, "realloc expand");
    for (int i = 0; i < 5; i++)
        printf("%d ", q[i]); // old data should be preserved
    printf("\n");
    free(q);
}

/*
 * Test 5: realloc failure simulation (huge size).
 */
void test_realloc_failure() {
    printf("Test 5: realloc failure\n");
    int *p = malloc(10 * sizeof(int));
    check_ptr(p, "malloc");
    int *q = realloc(p, (size_t)-1); // absurdly large
    if (!q) {
        printf("Realloc failed as expected, original block still valid\n");
        free(p);
    } else {
        free(q); // extremely unlikely
    }
}

/*
 * Test 6: realloc to the same size should be a no-op (implementation may return same or new pointer).
 */
void test_realloc_same_size() {
    printf("Test 6: realloc same size\n");
    int *p = malloc(10 * sizeof(int));
    check_ptr(p, "malloc");
    int *q = realloc(p, 10 * sizeof(int));
    check_ptr(q, "realloc same size");
    if (p == q)
        printf("Pointer unchanged\n");
    else
        printf("Pointer changed but valid\n");
    free(q);
}

int main(void) {
    test_realloc_null_malloc();
    test_realloc_zero();
    test_realloc_shrink();
    test_realloc_expand();
    test_realloc_failure();
    test_realloc_same_size();
    return 0;
}

