#include <limits.h>
#include <assert.h>

long nondet_long(void);

void __VERIFIER_error(void);
void reach_error() { __VERIFIER_error(); }
void nope() { ERROR: reach_error(); }

int main()
{
	long x = nondet_long();
	if (x > LONG_MIN)
		return 0;
	if (x == LONG_MIN)
		nope();
	return 0;
}
