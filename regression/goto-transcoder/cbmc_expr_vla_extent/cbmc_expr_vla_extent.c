int main() {
  unsigned n;
  __CPROVER_assume(n > 0 && n < 8);
  int v[n];
  for (unsigned i = 0; i < n; i++) v[i] = 3;
  __CPROVER_assert(v[0] == 3, "nondet-extent array element");
  return 0;
}
