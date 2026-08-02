// This file is part of the SV-Benchmarks collection of verification tasks:
// https://github.com/sosy-lab/sv-benchmarks
//
// SPDX-FileCopyrightText: 2025 RWTH Aachen
//
// SPDX-License-Identifier: LicenseRef-RWTH-Aachen

typedef enum {false,true} bool;

extern int __VERIFIER_nondet_int(void);

int main() {
    int i;
    i = __VERIFIER_nondet_int();
    
    while (i > 0) {
        if (i != 5) {
            i = i-1;
        }
    }
    
    return 0;
}
