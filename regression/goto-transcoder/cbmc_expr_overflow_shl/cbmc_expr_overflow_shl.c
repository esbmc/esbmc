int main() {
  int x = 1;
  __CPROVER_assert(!__CPROVER_overflow_shl(x, 3), "1<<3 does not overflow");
  __CPROVER_assert(__CPROVER_overflow_shl(x, 31), "1<<31 overflows int");
  return 0;
}
