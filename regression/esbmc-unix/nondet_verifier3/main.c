// This file is part of the SV-Benchmarks collection of verification tasks:
// https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks
//
// SPDX-FileCopyrightText: 2025 The SV-Benchmarks Community
//
// SPDX-License-Identifier: Apache-2.0

typedef long unsigned int size_t;

void reach_error() {}
extern void __VERIFIER_nondet_memory(void *ptr, size_t size);

struct A {
  int x;
  int y;
};

int main() {
  struct A a = {0, 0};
  __VERIFIER_nondet_memory(&a, sizeof(a));
  if (a.x != a.x || a.y != a.y) {
    reach_error();
  }
  return 0;
}
