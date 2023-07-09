/* ./esbmc --cheri purecap --force-malloc-success temporal.c */

#include <stdlib.h>
#include <assert.h>
#include <cheri/cheric.h>

int main() {
	int n = nondet_uint() % 1024; /* model arbitrary user input */
	char *__capability a = malloc_c(1+n); /* CHERI-C API */
	assert(cheri_gettag(a));              /* CHERI-C API */
	*a = 5;
	free_c(a);                            /* CHERI-C API */
	*a = 5; /* use-after-free bug */
}
