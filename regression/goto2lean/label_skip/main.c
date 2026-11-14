unsigned nondet_uint(void);
void v(unsigned c) {
  if (c) goto L;
  c = 2;
L:;
}
int main(void) {
  unsigned c = nondet_uint();
  while (c) { if (c == 3) continue; c = c - 1; }
  v(c);
  return 0;
}
