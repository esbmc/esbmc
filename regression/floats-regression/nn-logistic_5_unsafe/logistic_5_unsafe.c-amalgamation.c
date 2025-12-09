// This file is part of the SV-Benchmarks collection of verification tasks:
// https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
//
// SPDX-FileCopyrightText: 2023 Edoardo Manino
//
// SPDX-License-Identifier: MIT

#include <verifier_functions.h>

#include <math.h>

float logistic(float x)
{
	return 0.5f * tanhf(0.5f * x) + 0.5f;
}

#define LOGISTIC_CHECK_RANGE 10.0f
#define LOGISTIC_CHECK_ERROR 1e-6f

int main() /* check_inverse */
{
	float x = __VERIFIER_nondet_float();
	
	__VERIFIER_assume(isgreaterequal(x, -LOGISTIC_CHECK_RANGE) && islessequal(x, LOGISTIC_CHECK_RANGE)); /* Choose a range where precision is high */
	
	float y = 2.0 * atanhf(2.0f * logistic(x) - 1);
	float z = fabsf(x - y);
	
	__VERIFIER_assert(islessequal(z, LOGISTIC_CHECK_ERROR)); /* Expected result: verification failure */

    return 0;
}
