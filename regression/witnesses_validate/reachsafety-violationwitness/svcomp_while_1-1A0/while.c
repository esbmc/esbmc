// This file is part of the SV-Benchmarks collection of verification tasks:
// https://github.com/sosy-lab/sv-benchmarks
//
// SPDX-FileCopyrightText: 2024 Paulína Ayaziová
//
// SPDX-License-Identifier: GPL-3.0-or-later

extern void __assert_fail(const char *, const char *, unsigned int, const char *) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__noreturn__));
void reach_error(){ __assert_fail("0", "while.c", 2, "reach_error"); }

extern int __VERIFIER_nondet_int();

int main() {
            
    int count = 0;    
    int a = __VERIFIER_nondet_int();

    if (a <= 0)
        return 0;

    while (a > 0) {
        int n = __VERIFIER_nondet_int();
        if (n > 0)
            a = a - n;
        count++;
    }

    if (count == -a)
        reach_error();

    return 0;
}

