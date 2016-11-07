#include <assert.h>
#include <math.h>

float nondet_float (void);

int main (void) {
	float f = nondet_float();
        // Everything in this range hits the bug
	__ESBMC_assume(f < +INFINITY);
	__ESBMC_assume(f >= 0x1.0p+107f);
	__ESBMC_assume(f > 0.0f);

	float g = nondet_float();
	__ESBMC_assume(g <= 0x1.0p-149f);
	__ESBMC_assume(g > 0.0f);

	// exponent is >= 107 - -127 = 234 = 255 - 21
	float div = f / g;

	assert(!(div == 0.0));

	return 1;
}
