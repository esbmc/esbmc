unsigned nondet_uint(void);
int main(void) {
  unsigned c = nondet_uint(), x = 0;
  goto M;
L:
  x = x + 1;
M:
  if (x < c) goto L;
  return 0;
}
