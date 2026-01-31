// Simulates an operational model header WITHOUT __ESBMC_EXTERN_NOVAL.
//
// In ESBMC's operational models (e.g., time.h), headers declare extern
// variables that are defined in the corresponding library (e.g., time.c).
// The library is compiled into ESBMC and linked internally.
//
// Problem: Without __ESBMC_EXTERN_NOVAL, ESBMC assigns nondet values to
// extern declarations. When the operational model library is linked
// internally, these nondet values can clash with the actual definitions.
//
// Solution: Use __ESBMC_EXTERN_NOVAL on extern declarations in operational
// model headers. This leaves the value as nil, allowing the library's
// definition to take effect during ESBMC's internal linking.
//
// This test demonstrates the failure case: the extern gets a nondet value,
// causing assertions about the expected defined value to fail.

extern int counter;
