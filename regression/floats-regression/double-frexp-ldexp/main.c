#include <math.h>
#include <assert.h>
#include <float.h>  /* DBL_TRUE_MIN */

int main()
{
	double x = nondet_double();
	int xe = nondet_int();
	double xm = frexp(x, &xe);
	assert(isfinite(xm) == isfinite(x));
	assert(isinf(xm) == isinf(x));
	assert(isnan(xm) == isnan(x));
	if (!isnan(x)) {
		double x2 = ldexp(xm, xe);
		assert(x2 == x);
		assert(signbit(x) == signbit(xm));
	}
	if (isfinite(x) && x > 0.0) {
		assert(0.5 <= xm);
		assert(xm < 1.0);
		assert(xe >= -1022 - 52);
		assert(xe <=  1024);
		assert((fpclassify(x) == FP_SUBNORMAL) == (xe < DBL_MIN_EXP));
	}

	double y = nondet_double();
	int ye;
	double ym = frexp(y, &ye);
	__ESBMC_assume(ym == 0.0 || (ym >= 0.5 && ym < 1));
	assert(isfinite(y));

	int ze;
	frexp(DBL_MIN, &ze);
	assert(ze == DBL_MIN_EXP);

#if __STDC_VERSION__ >= 201112L && __DBL_HAS_DENORM__
	frexp(DBL_TRUE_MIN, &ze);
	assert(ze == DBL_MIN_EXP - (DBL_MANT_DIG - 1));
#endif
	frexp(0.0, &ze);
	assert(ze == 0);

	frexp(0x1p42, &ze);
	assert(ze == 43);

	frexp(0x1p-42, &ze);
	assert(ze == -41);

	frexp(DBL_MAX, &ze);
	assert(ze == DBL_MAX_EXP);

	double r = frexp(0.1234567890123456789, &ze);
	assert(r == 0x1.f9add3746f65fp-1);
	assert(ze == -3);

	r = frexp(0.9999999999999999, &ze);
	assert(r == 0x1.fffffffffffffp-1);
	assert(ze == 0);

	r = frexp(-1, &ze);
	assert(r == -.5);
	assert(ze == 1);

	r = frexp(.5, &ze);
	assert(r == .5);
	assert(ze == 0);
}
