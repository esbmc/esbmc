#include <stdlib.h>

extern _Bool __VERIFIER_nondet_bool();

/* simple regression test for atexit */

int **g = NULL;

void free_g1() {
	free(g);
	g = NULL;
}

void free_g2() {
	if (g != NULL)
		free(*g);
}

void h() {
	exit(1); // memory leak, but still reachable
}

void f() {
	*g = (int *) malloc(sizeof(int));
	atexit(free_g2);
	h();
}


int main() {
	g = (int **) malloc(sizeof(int *));
// 	atexit(free_g1);
	f();
	free(*g);
	free(g);
	g = NULL;
	return 0;
}
