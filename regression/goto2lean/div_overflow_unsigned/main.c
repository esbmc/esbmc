unsigned nondet_uint(void);
int main(void) {
  unsigned x = nondet_uint(), y = nondet_uint();
  unsigned q = x / y;
  return 0;
}
