#include <stdlib.h>
#include <stdio.h>

extern _Bool __VERIFIER_nondet_bool();

/* Enhanced regression test for atexit with loops */

int **g = NULL;
int *h_ptr = NULL;
int *array = NULL;

void free_g1() {
    printf("Freeing g\n");
    free(g);
    g = NULL;
}

void free_g2() {
    if (g != NULL) {
        printf("Freeing *g\n");
        free(*g);
    }
}

void free_h_ptr() {
    printf("Freeing h_ptr\n");
    free(h_ptr);
    h_ptr = NULL;
}

void free_array() {
    printf("Freeing array\n");
    free(array);
    array = NULL;
}

void extra_cleanup() {
    printf("Performing extra cleanup\n");
    if (g != NULL && *g != NULL) {
        **g = 0; // Just to show additional manipulation before freeing
    }
    if (array != NULL) {
        for (int i = 0; i < 4; i++) {
            array[i] = 0; // Reset values before freeing
        }
    }
}

void h() {
    for (int i = 0; i < 4; i++) {
        if (__VERIFIER_nondet_bool()) {
            printf("Exiting from h() in loop iteration %d\n", i);
            exit(1);
        }
    }
}

void f() {
    *g = (int *)malloc(sizeof(int));
    if (*g == NULL) {
        perror("malloc failed for *g");
        exit(1);
    }
    **g = 42; // Assign a value
    atexit(free_g2);
    
    h_ptr = (int *)malloc(sizeof(int));
    if (h_ptr == NULL) {
        perror("malloc failed for h_ptr");
        exit(1);
    }
    *h_ptr = 99;
    atexit(free_h_ptr);
    
    array = (int *)malloc(4 * sizeof(int));
    if (array == NULL) {
        perror("malloc failed for array");
        exit(1);
    }
    for (int i = 0; i < 4; i++) {
        array[i] = i * 2;
    }
    atexit(free_array);
    
    atexit(extra_cleanup);
    h();
}

int main() {
    g = (int **)malloc(sizeof(int *));
    if (g == NULL) {
        perror("malloc failed for g");
        exit(1);
    }
    atexit(free_g1);
    
    for (int i = 0; i < 3; i++) {
        if (__VERIFIER_nondet_bool()) {
            printf("Exiting from main() in loop iteration %d before f()\n", i);
            exit(1);
        }
    }
    
    f();
    return 0;
}


