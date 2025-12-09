#include <limits.h>
#include <assert.h>

unsigned long nondet_ulong(void);

void __VERIFIER_error(void);
void reach_error() { __VERIFIER_error(); }
void nope() { ERROR: reach_error(); }

int main()
{
	unsigned long x = nondet_ulong();
	if (x < ULONG_MAX)
		return 0;
	if (x == ULONG_MAX)
		nope();
	return 0;
}
