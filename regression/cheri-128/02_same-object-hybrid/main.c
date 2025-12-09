/* ./esbmc --cheri hybrid assume3.c --z3 */

#include <stdlib.h>
#include <assert.h>
#include <cheri/cheric.h>

int main() {
	int n = nondet_uint() % 1024; /* model arbitrary user input */
	char a[n+1];
	a[n] = 17; /* succeeds */
	char *__capability b = a;
	char *__capability c = cheri_setbounds(b-1, n); /* fails: not the same object */
}
