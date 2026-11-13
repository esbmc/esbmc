int nondet_int(void);
int main(void) {
  int x = nondet_int();
  int z = x << 1;
  return 0;
}
