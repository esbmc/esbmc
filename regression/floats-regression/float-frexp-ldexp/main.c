#include <math.h>
#include <assert.h>
#include <float.h>  /* FLT_TRUE_MIN */

int main()
{
	float x = nondet_float();
	int xe = nondet_int();
	float xm = frexpf(x, &xe);
	assert(isfinite(xm) == isfinite(x));
	assert(isinf(xm) == isinf(x));
	assert(isnan(xm) == isnan(x));
	if (!isnan(x)) {
		assert(ldexpf(xm, xe) == x);
		assert(signbit(x) == signbit(xm));
	}
	if (isfinite(x) && x > 0.0) {
		assert(0.5 <= xm);
		assert(xm < 1.0);
		assert(xe >= -1022 - 52);
		assert(xe <=  1024);
		assert((fpclassify(x) == FP_SUBNORMAL) == (xe < FLT_MIN_EXP));
	}

	float y = nondet_float();
	int ye;
	float ym = frexpf(y, &ye);
	__ESBMC_assume(ym == 0.0 || (ym >= 0.5 && ym < 1));
	assert(isfinite(y));

	int ze;
	frexpf(FLT_MIN, &ze);
	assert(ze == FLT_MIN_EXP);

#if __STDC_VERSION__ >= 201112L && __FLT_HAS_DENORM__
	frexpf(FLT_TRUE_MIN, &ze);
	assert(ze == FLT_MIN_EXP - (FLT_MANT_DIG - 1));
#endif
	frexpf(0.0, &ze);
	assert(ze == 0);

	frexpf(0x1p42, &ze);
	assert(ze == 43);

	frexpf(0x1p-42, &ze);
	assert(ze == -41);

	frexpf(FLT_MAX, &ze);
	assert(ze == FLT_MAX_EXP);
}
