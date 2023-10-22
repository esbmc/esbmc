#include <math.h>
#include <assert.h>

int main()
{
	double x = nondet_double();
	int xe = nondet_int();
	double xm = frexp(x, &xe);
	assert(isfinite(xm) == isfinite(x));
	assert(isinf(xm) == isinf(x));
	assert(isnan(xm) == isnan(x));
	if (!isnan(x)) {
		assert(ldexp(xm, xe) == x);
		assert(signbit(x) == signbit(xm));
	}
	if (isfinite(x) && x > 0.0) {
		assert(0.5 <= xm);
		assert(xm < 1.0);
		assert(xe >= -1022 - 52);
		assert(xe <=  1024);
		assert((fpclassify(x) == FP_SUBNORMAL) == (xe <= -1022));
	}

	double y = nondet_double();
	int ye;
	double ym = frexp(y, &ye);
	__ESBMC_assume(ym == 0.0 || (ym >= 0.5 && ym < 1));
	assert(isfinite(y));
}
