// This file is part of the SV-Benchmarks collection of verification tasks:
// https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
//
// SPDX-FileCopyrightText: 2023 Edoardo Manino
//
// SPDX-License-Identifier: MIT

#include <verifier_functions.h>

#include <math.h>

#define LOG_CHECK_NEXT 1e-2f

int main() /* check_derivative */
{
	float x1 = __VERIFIER_nondet_float();
	float x2 = x1 + LOG_CHECK_NEXT;
	
	__VERIFIER_assume(isgreaterequal(x1, 0.0f) && !isinf(x1));
	
	float y1 = logf(x1);
	float y2 = logf(x2);
	float derivative = (y2 - y1) / LOG_CHECK_NEXT;
	
	float d1 = 1.0f / x1; /* analytical derivative of log(x) at x = x1 */
	
	__VERIFIER_assert(isgreaterequal(derivative, d1)); /* Expected result: verification failure */

    return 0;
}
