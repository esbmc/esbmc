#include <stdlib.h>
#include <assert.h>

void check_ptr(void *ptr) {
    assert(ptr != NULL); // simplify: no perror/exit
}

/*
 * Test 1: Dynamic vector growth using realloc
 * Verify that elements already inserted remain correct.
 */
void test_vector_growth() {
    int capacity = 1;
    int size = 0;
    int *vec = malloc(capacity * sizeof(int));
    check_ptr(vec);

    for (int i = 0; i < 20; i++) {
        if (size == capacity) {
            capacity *= 2;
            int *tmp = realloc(vec, capacity * sizeof(int));
            check_ptr(tmp);
            vec = tmp;
        }
        vec[size++] = i;

        // assert data integrity
        for (int j = 0; j < size; j++) {
            assert(vec[j] == j);
        }
    }

    free(vec);
}

/*
 * Test 2: Shrinking vector
 * Verify last element is preserved after each shrink.
 */
void test_vector_shrink() {
    int capacity = 16;
    int size = capacity;
    int *vec = malloc(capacity * sizeof(int));
    check_ptr(vec);

    for (int i = 0; i < size; i++) vec[i] = i;

    for (int new_cap = capacity; new_cap > 0; new_cap /= 2) {
        int *tmp = realloc(vec, new_cap * sizeof(int));
        check_ptr(tmp);
        vec = tmp;
        size = new_cap;

        assert(vec[size - 1] == size - 1);
    }

    free(vec);
}

/*
 * Test 3: Alternating shrink and expand
 * Verify that old contents are intact after every resize.
 */
void test_alternating_resize() {
    int n = 8;
    int *arr = malloc(n * sizeof(int));
    check_ptr(arr);

    for (int i = 0; i < n; i++) arr[i] = i;

    for (int round = 0; round < 3; round++) {
        // shrink
        int shrink_n = n / 2;
        int *tmp = realloc(arr, shrink_n * sizeof(int));
        check_ptr(tmp);
        arr = tmp;
        for (int i = 0; i < shrink_n; i++)
            assert(arr[i] == i);
        n = shrink_n;

        // expand
        int expand_n = n * 3;
        tmp = realloc(arr, expand_n * sizeof(int));
        check_ptr(tmp);
        arr = tmp;
        for (int i = 0; i < n; i++)
            assert(arr[i] == i);
        for (int i = n; i < expand_n; i++)
            arr[i] = i;
        n = expand_n;
        assert(arr[n - 1] == n - 1);
    }

    free(arr);
}

/*
 * Test 4: Resize to zero, then grow again
 */
void test_resize_zero_grow() {
    int *arr = malloc(5 * sizeof(int));
    check_ptr(arr);
    for (int i = 0; i < 5; i++) arr[i] = i;

    int *tmp = realloc(arr, 0);
    arr = tmp; // freed, likely NULL

    arr = realloc(NULL, 10 * sizeof(int));
    check_ptr(arr);
    for (int i = 0; i < 10; i++) arr[i] = i * 2;

    assert(arr[0] == 0);
    assert(arr[9] == 18);

    free(arr);
}

int main(void) {
    test_vector_growth();
    test_vector_shrink();
    test_alternating_resize();
    test_resize_zero_grow();
    return 0;
}

