
#include <assert.h>
#include <limits.h>
#include <stdlib.h>

int main()
{
	unsigned n = nondet_uint();
	__ESBMC_assume(n);
	char *s = malloc(n);
	__ESBMC_assume(!isspace(s[0]));
	s[n-1] = '\0';

	int ai = atoi(s);
	long al = atol(s);
	long long all = atoll(s);

	long sl = strtol(s, NULL, 10);
	// long long sll = strtoll(s, NULL, 10);

	assert(ai == al || ai == INT_MIN || ai == INT_MAX);
	assert(al == sl);
	// assert(all == sll);
}
