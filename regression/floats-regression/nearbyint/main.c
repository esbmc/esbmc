#include <fenv.h>
#include <math.h>

extern void __VERIFIER_error() __attribute__ ((__noreturn__));
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: __VERIFIER_error(); } return; }

int main(void) {
	int const modes[] = {FE_DOWNWARD, FE_TONEAREST, FE_TOWARDZERO, FE_UPWARD};

	for (int i = 0; i < sizeof(modes) / sizeof(int); ++i) {
		fesetround(modes[i]);
		switch (modes[i]) {
		case FE_DOWNWARD:
			__VERIFIER_assert(nearbyint(12.9) == 12.);
			__VERIFIER_assert(nearbyint(-12.1) == -13.);
			break;
		case FE_TONEAREST:
			__VERIFIER_assert(nearbyint(12.4) == 12.);
			__VERIFIER_assert(nearbyint(-12.4) == -12.);
			break;
		case FE_TOWARDZERO:
			__VERIFIER_assert(nearbyint(12.9) == 12.);
			__VERIFIER_assert(nearbyint(-12.9) == -12.);
			break;
		case FE_UPWARD:
			__VERIFIER_assert(nearbyint(12.1) == 13.);
			__VERIFIER_assert(nearbyint(-12.9) == -12.);
			break;
		}
	}

	fesetround(FE_DOWNWARD);
	__VERIFIER_assert(nearbyint(12.9) == 12.);
	__VERIFIER_assert(nearbyint(-12.1) == -13.);
	fesetround(FE_TONEAREST);
	__VERIFIER_assert(nearbyint(12.4) == 12.);
	__VERIFIER_assert(nearbyint(-12.4) == -12.);
	fesetround(FE_TOWARDZERO);
	__VERIFIER_assert(nearbyint(12.9) == 12.);
	__VERIFIER_assert(nearbyint(-12.9) == -12.);
	fesetround(FE_UPWARD);
	__VERIFIER_assert(nearbyint(12.1) == 13.);
	__VERIFIER_assert(nearbyint(-12.9) == -12.);
	return 0;
}

