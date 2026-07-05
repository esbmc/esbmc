// This file is part of the SV-Benchmarks collection of verification tasks:
// https://github.com/sosy-lab/sv-benchmarks
//
// SPDX-FileCopyrightText: 2025 The SV-Benchmarks Community
//
// SPDX-License-Identifier: GPL-3.0-or-later

extern int __VERIFIER_nondet_int();

int main() {

    int x = __VERIFIER_nondet_int();
    int y = __VERIFIER_nondet_int();
    int tmp;

    while(x != y) {
        tmp = x;
        x = y;
        y = tmp;

        if (__VERIFIER_nondet_int())
            break;
    }

    return 0;
}
