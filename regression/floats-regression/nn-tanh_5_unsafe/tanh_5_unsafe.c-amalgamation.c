// This file is part of the SV-Benchmarks collection of verification tasks:
// https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
//
// SPDX-FileCopyrightText: 2023 Edoardo Manino
//
// SPDX-License-Identifier: MIT

#include <verifier_functions.h>

#include <math.h>

#define TANH_CHECK_RANGE 5.0f
#define TANH_CHECK_ERROR 1e-6f

int main() /* check_inverse */
{
	float x = __VERIFIER_nondet_float();
	
	__VERIFIER_assume(isgreaterequal(x, -TANH_CHECK_RANGE) && islessequal(x, TANH_CHECK_RANGE)); /* Choose a range where precision is high */
	
	float y = atanhf(tanhf(x));
	float z = fabsf(x - y);
	
	__VERIFIER_assert(islessequal(z, TANH_CHECK_ERROR)); /* Expected result: verification failure */

    return 0;
}
