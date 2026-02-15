/*
 * Memory Safety Verification Example
 *
 * Demonstrates ESBMC's memory safety checking capabilities:
 * - Buffer overflow detection
 * - Null pointer dereference
 * - Memory leak detection
 * - Use-after-free detection
 *
 * Run with: esbmc memory-check.c --memory-leak-check --unwind 10
 */

#include <stdlib.h>
#include <assert.h>

// Non-deterministic input for symbolic verification
int __ESBMC_nondet_int(void);
void __ESBMC_assume(_Bool);
void __ESBMC_assert(_Bool, const char *);

/* Example 1: Buffer Overflow Detection */
void buffer_overflow_example(void) {
    int arr[10];
    int idx = __ESBMC_nondet_int();

    // Without this assumption, ESBMC will find the buffer overflow
    // Uncomment to make verification pass:
    // __ESBMC_assume(idx >= 0 && idx < 10);

    // ESBMC will detect if idx can be out of bounds
    arr[idx] = 42;
}

/* Example 2: Safe Array Access */
void safe_array_access(void) {
    int arr[10];
    int idx = __ESBMC_nondet_int();

    // Properly constrained index
    __ESBMC_assume(idx >= 0 && idx < 10);

    // This is now safe
    arr[idx] = 42;

    // Verify the value was written
    __ESBMC_assert(arr[idx] == 42, "Value correctly written");
}

/* Example 3: Null Pointer Check */
int *create_array(int size) {
    if (size <= 0) {
        return NULL;
    }
    return (int *)malloc(size * sizeof(int));
}

void null_pointer_example(void) {
    int size = __ESBMC_nondet_int();
    int *arr = create_array(size);

    // Without null check, ESBMC detects potential null dereference
    // Uncomment to fix:
    // if (arr == NULL) return;

    arr[0] = 10;  // Potential null dereference

    free(arr);
}

/* Example 4: Safe Null Handling */
void safe_null_handling(void) {
    int size = __ESBMC_nondet_int();
    __ESBMC_assume(size > 0 && size <= 100);

    int *arr = create_array(size);

    if (arr != NULL) {
        arr[0] = 10;
        __ESBMC_assert(arr[0] == 10, "Value set correctly");
        free(arr);
    }
}

/* Example 5: Memory Leak Detection */
void memory_leak_example(void) {
    int *ptr = (int *)malloc(sizeof(int));
    *ptr = 42;

    // Memory leak: ptr is never freed
    // ESBMC with --memory-leak-check will detect this
}

/* Example 6: Proper Memory Management */
void proper_memory_management(void) {
    int *ptr = (int *)malloc(sizeof(int));

    if (ptr != NULL) {
        *ptr = 42;
        __ESBMC_assert(*ptr == 42, "Value stored correctly");
        free(ptr);  // Properly freed
    }
}

/* Example 7: Double Free Detection */
void double_free_example(void) {
    int *ptr = (int *)malloc(sizeof(int));

    if (ptr != NULL) {
        *ptr = 10;
        free(ptr);
        // free(ptr);  // Double free - uncomment to see ESBMC detect it
    }
}

/* Example 8: Use After Free Detection */
void use_after_free_example(void) {
    int *ptr = (int *)malloc(sizeof(int));

    if (ptr != NULL) {
        *ptr = 10;
        free(ptr);
        // int x = *ptr;  // Use after free - uncomment to see detection
    }
}

/* Example 9: Dynamic Array with Bounds Checking */
void dynamic_array_bounds(void) {
    int size = __ESBMC_nondet_int();
    __ESBMC_assume(size > 0 && size <= 50);

    int *arr = (int *)malloc(size * sizeof(int));
    __ESBMC_assume(arr != NULL);

    int idx = __ESBMC_nondet_int();
    __ESBMC_assume(idx >= 0 && idx < size);

    // Safe access within bounds
    arr[idx] = 100;
    __ESBMC_assert(arr[idx] == 100, "Dynamic array access correct");

    free(arr);
}

/* Example 10: Struct Memory Safety */
typedef struct {
    int *data;
    int size;
} Container;

Container *create_container(int size) {
    if (size <= 0) return NULL;

    Container *c = (Container *)malloc(sizeof(Container));
    if (c == NULL) return NULL;

    c->data = (int *)malloc(size * sizeof(int));
    if (c->data == NULL) {
        free(c);
        return NULL;
    }

    c->size = size;
    return c;
}

void destroy_container(Container *c) {
    if (c != NULL) {
        free(c->data);
        free(c);
    }
}

void container_example(void) {
    int size = __ESBMC_nondet_int();
    __ESBMC_assume(size > 0 && size <= 20);

    Container *c = create_container(size);

    if (c != NULL) {
        // Safe access
        int idx = __ESBMC_nondet_int();
        __ESBMC_assume(idx >= 0 && idx < c->size);

        c->data[idx] = 42;
        __ESBMC_assert(c->data[idx] == 42, "Container data access correct");

        destroy_container(c);
    }
}

int main(void) {
    // Comment/uncomment to test different examples

    // Examples with bugs (will fail verification):
    // buffer_overflow_example();
    // null_pointer_example();
    // memory_leak_example();

    // Examples that pass verification:
    safe_array_access();
    safe_null_handling();
    proper_memory_management();
    dynamic_array_bounds();
    container_example();

    return 0;
}
