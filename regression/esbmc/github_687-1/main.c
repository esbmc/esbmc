#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>

// Struct definitions
typedef struct {
} __VerificationLoopDS;
typedef struct {
	float x1;
	float x2;
	float x3;
	float x4;
	float x5;
	float x6;
	float x7;
	float x8;
	float x9;
	float x10;
	float x11;
	float x12;
	float x13;
	bool z1;
	bool z2;
	bool z3;
	bool z4;
	bool z5;
	bool z6;
	bool z7;
	bool z8;
	bool z9;
	bool z10;
	bool z11;
	bool z12;
	bool z13;
} __real_if;
typedef struct {
	float x;
	bool z;
} __real_if_function;
// Global variables
__real_if_function real_if_function1;
__real_if instance;
__real_if_function real_if_function1_inlined_1;
__real_if_function real_if_function1_inlined_2;
__real_if_function real_if_function1_inlined_3;
__real_if_function real_if_function1_inlined_4;
__real_if_function real_if_function1_inlined_5;
__real_if_function real_if_function1_inlined_6;
__real_if_function real_if_function1_inlined_7;
__real_if_function real_if_function1_inlined_8;
__real_if_function real_if_function1_inlined_9;
__real_if_function real_if_function1_inlined_10;
__real_if_function real_if_function1_inlined_11;
__real_if_function real_if_function1_inlined_12;
__real_if_function real_if_function1_inlined_13;
uint16_t __assertion_error;
__VerificationLoopDS verificationLoop;
bool __cbmc_eoc_marker;
// Forward declarations of the generated functions
void real_if_function(__real_if_function *__context);
void real_if(__real_if *__context);
void VerificationLoop();
// Declare nondet assignment functions
_Bool __VERIFIER_nondet_bool();
uint8_t __VERIFIER_nondet_uint8_t(void);
uint16_t __VERIFIER_nondet_uint16_t(void);
uint32_t __VERIFIER_nondet_uint32_t(void);
uint64_t __VERIFIER_nondet_uint64_t(void);
int8_t __VERIFIER_nondet_int8_t(void);
int16_t __VERIFIER_nondet_int16_t(void);
int32_t __VERIFIER_nondet_int32_t(void);
int64_t __VERIFIER_nondet_int64_t(void);
float __VERIFIER_nondet_float(void);
double __VERIFIER_nondet_double(void);
// Translated functions
void real_if_function(__real_if_function *__context) {
	// Temporary variables
	{
		if ((__context->x > 10.0)) {
			__context->z = true;
		}
		else {
			__context->z = false;
		}
		return;
	}
}
void real_if(__real_if *__context) {
	// Temporary variables
	{
		// Assign inputs
		real_if_function1_inlined_1.x = __context->x1;
		real_if_function(&real_if_function1_inlined_1);
		// Assign outputs
		__context->z1 = real_if_function1_inlined_1.z;
		// Assign inputs
		real_if_function1_inlined_2.x = __context->x2;
		real_if_function(&real_if_function1_inlined_2);
		// Assign outputs
		__context->z2 = real_if_function1_inlined_2.z;
		// Assign inputs
		real_if_function1_inlined_3.x = __context->x3;
		real_if_function(&real_if_function1_inlined_3);
		// Assign outputs
		__context->z3 = real_if_function1_inlined_3.z;
		// Assign inputs
		real_if_function1_inlined_4.x = __context->x4;
		real_if_function(&real_if_function1_inlined_4);
		// Assign outputs
		__context->z4 = real_if_function1_inlined_4.z;
		// Assign inputs
		real_if_function1_inlined_5.x = __context->x5;
		real_if_function(&real_if_function1_inlined_5);
		// Assign outputs
		__context->z5 = real_if_function1_inlined_5.z;
		// Assign inputs
		real_if_function1_inlined_6.x = __context->x6;
		real_if_function(&real_if_function1_inlined_6);
		// Assign outputs
		__context->z6 = real_if_function1_inlined_6.z;
		// Assign inputs
		real_if_function1_inlined_7.x = __context->x7;
		real_if_function(&real_if_function1_inlined_7);
		// Assign outputs
		__context->z7 = real_if_function1_inlined_7.z;
		// Assign inputs
		real_if_function1_inlined_8.x = __context->x8;
		real_if_function(&real_if_function1_inlined_8);
		// Assign outputs
		__context->z8 = real_if_function1_inlined_8.z;
		// Assign inputs
		real_if_function1_inlined_9.x = __context->x9;
		real_if_function(&real_if_function1_inlined_9);
		// Assign outputs
		__context->z9 = real_if_function1_inlined_9.z;
		// Assign inputs
		real_if_function1_inlined_10.x = __context->x10;
		real_if_function(&real_if_function1_inlined_10);
		// Assign outputs
		__context->z10 = real_if_function1_inlined_10.z;
		// Assign inputs
		real_if_function1_inlined_11.x = __context->x11;
		real_if_function(&real_if_function1_inlined_11);
		// Assign outputs
		__context->z11 = real_if_function1_inlined_11.z;
		// Assign inputs
		real_if_function1_inlined_12.x = __context->x12;
		real_if_function(&real_if_function1_inlined_12);
		// Assign outputs
		__context->z12 = real_if_function1_inlined_12.z;
		// Assign inputs
		real_if_function1_inlined_13.x = __context->x13;
		real_if_function(&real_if_function1_inlined_13);
		// Assign outputs
		__context->z13 = real_if_function1_inlined_13.z;
		if ((! ((((((((((((__context->z1 && __context->z2) && __context->z3) && __context->z4) && __context->z5) && __context->z6) && __context->z7) && __context->z8) && __context->z9) && __context->z10) && __context->z11) && __context->z12) && __context->z13))) {
			__assertion_error = 1;
		}
		return;
	}
}
void VerificationLoop() {
	// Temporary variables
	{
		while (true) {
			instance.x1 = __VERIFIER_nondet_float();
			instance.x10 = __VERIFIER_nondet_float();
			instance.x11 = __VERIFIER_nondet_float();
			instance.x12 = __VERIFIER_nondet_float();
			instance.x13 = __VERIFIER_nondet_float();
			instance.x2 = __VERIFIER_nondet_float();
			instance.x3 = __VERIFIER_nondet_float();
			instance.x4 = __VERIFIER_nondet_float();
			instance.x5 = __VERIFIER_nondet_float();
			instance.x6 = __VERIFIER_nondet_float();
			instance.x7 = __VERIFIER_nondet_float();
			instance.x8 = __VERIFIER_nondet_float();
			instance.x9 = __VERIFIER_nondet_float();
			// Assign inputs
			real_if(&instance);
			// Assign outputs
			VerificationLoop_prepare_EoC: {
				assert((__assertion_error == 0));
				__cbmc_eoc_marker = true; // to indicate the end of the loop for the counterexample parser
				__cbmc_eoc_marker = false;
			}
		}
		return;
	}
}
// Entry point
int main(void) {
	// Initial values
	real_if_function1.x = 0.0;
	real_if_function1.z = false;
	instance.x1 = 0.0;
	instance.x2 = 0.0;
	instance.x3 = 0.0;
	instance.x4 = 0.0;
	instance.x5 = 0.0;
	instance.x6 = 0.0;
	instance.x7 = 0.0;
	instance.x8 = 0.0;
	instance.x9 = 0.0;
	instance.x10 = 0.0;
	instance.x11 = 0.0;
	instance.x12 = 0.0;
	instance.x13 = 0.0;
	instance.z1 = false;
	instance.z2 = false;
	instance.z3 = false;
	instance.z4 = false;
	instance.z5 = false;
	instance.z6 = false;
	instance.z7 = false;
	instance.z8 = false;
	instance.z9 = false;
	instance.z10 = false;
	instance.z11 = false;
	instance.z12 = false;
	instance.z13 = false;
	real_if_function1_inlined_1.x = 0.0;
	real_if_function1_inlined_1.z = false;
	real_if_function1_inlined_2.x = 0.0;
	real_if_function1_inlined_2.z = false;
	real_if_function1_inlined_3.x = 0.0;
	real_if_function1_inlined_3.z = false;
	real_if_function1_inlined_4.x = 0.0;
	real_if_function1_inlined_4.z = false;
	real_if_function1_inlined_5.x = 0.0;
	real_if_function1_inlined_5.z = false;
	real_if_function1_inlined_6.x = 0.0;
	real_if_function1_inlined_6.z = false;
	real_if_function1_inlined_7.x = 0.0;
	real_if_function1_inlined_7.z = false;
	real_if_function1_inlined_8.x = 0.0;
	real_if_function1_inlined_8.z = false;
	real_if_function1_inlined_9.x = 0.0;
	real_if_function1_inlined_9.z = false;
	real_if_function1_inlined_10.x = 0.0;
	real_if_function1_inlined_10.z = false;
	real_if_function1_inlined_11.x = 0.0;
	real_if_function1_inlined_11.z = false;
	real_if_function1_inlined_12.x = 0.0;
	real_if_function1_inlined_12.z = false;
	real_if_function1_inlined_13.x = 0.0;
	real_if_function1_inlined_13.z = false;
	__assertion_error = 0;
	// Custom entry logic
	VerificationLoop();
	return 0;
}
