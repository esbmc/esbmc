/* ./esbmc --cheri purecap --force-malloc-success assume.c --z3 */

#include <stdlib.h>
#include <assert.h>
#include <cheri/cheric.h>

int main() {
	int n = nondet_uint() % 1024; /* model arbitrary user input */
	char a[n+1];
	a[n] = 17; /* succeeds */
	__ESBMC_assume(cheri_getlength(a) == n);
	a[n] = 23; /* succeeds - vacuous truth */
}
