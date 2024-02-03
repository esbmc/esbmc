// This file is part of the SV-Benchmarks collection of verification tasks:
// https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
//
// SPDX-FileCopyrightText: 2023 Edoardo Manino
//
// SPDX-License-Identifier: MIT

#ifndef _VERIFICATION_FUNCTIONS_H
#define _VERIFICATION_FUNCTIONS_H

// Functions for verification (harness).
// Mainly consists of assume, reach_error, and nondets
// See: https://sv-comp.sosy-lab.org/2023/rules.php
#include <assert.h>
float __VERIFIER_nondet_float();
void reach_error() {
    assert(0);
}

#include <stdlib.h>

#define __VERIFIER_assume(cond) if(!(cond)) abort()
#define __VERIFIER_assert(cond) if(!(cond)) reach_error()

#endif
