int main() {
  unsigned u = 0x80000000u;
  int s = -8;
  __CPROVER_assert((u >> 4) == 0x18000000u, "lshr wrongly sign-extended");
  __CPROVER_assert((s >> 2) == -2, "ashr is arithmetic");
  return 0;
}
