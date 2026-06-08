int main() {
  int y = nondet_int();
  __ESBMC_assume(y>=0 && y <=10);
  while (y >= 0 && y <= 10) {
    y = (2*y + 1) / 2;
  }     
  return 0;
}
