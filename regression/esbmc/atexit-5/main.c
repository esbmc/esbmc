#include <stdlib.h>
#include <stdio.h>

extern _Bool __VERIFIER_nondet_bool();

/* Enhanced regression test for atexit */

int **g = NULL;
int *h_ptr = NULL;

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

void extra_cleanup() {
    printf("Performing extra cleanup\n");
    if (g != NULL && *g != NULL) {
        **g = 0; // Just to show additional manipulation before freeing
    }
}

void h() {
    if (__VERIFIER_nondet_bool()) {
        printf("Exiting from h()\n");
        exit(1);
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

    if (__VERIFIER_nondet_bool()) {
        printf("Exiting from main() before f()\n");
        exit(1);
    }

    f();
    return 0;
}

