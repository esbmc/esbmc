#include <math.h>

int main()
{
    double x = nondet_float();

	__ESBMC_assume(isgreaterequal(x, 0.0));
	
	__ESBMC_assert(x == x, "");

    return 0;
}
