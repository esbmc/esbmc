// This file is part of the SV-Benchmarks collection of verification tasks:
// https://github.com/sosy-lab/sv-benchmarks
//
// SPDX-FileCopyrightText: 2024 Paulína Ayaziová
//
// SPDX-License-Identifier: GPL-3.0-or-later

extern void __assert_fail(const char *, const char *, unsigned int, const char *) __attribute__ ((__nothrow__, __leaf__)) __attribute__ ((__noreturn__));
void reach_error(){ __assert_fail("0", "for.c", 2, "reach_error"); }

extern char __VERIFIER_nondet_char();
extern unsigned int __VERIFIER_nondet_uint();

int main() {
    char a[20];
    
    unsigned int count = 0;    
    for (int i = 0; i < 20; i++) {
        a[i] =  __VERIFIER_nondet_char();
        if (a[i] == 'a')
            count++;
    }

    if (count == 0)
        reach_error();

    if (count == __VERIFIER_nondet_uint())
        reach_error();

    return 0;
}

