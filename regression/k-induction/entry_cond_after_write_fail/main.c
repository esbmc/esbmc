// The x == 5 test runs after x++, so assuming it at the loop head left out
// the x = 50 path.
int nondet_int();
int main() {
  int x = 0;
  while (1) {
  while_continue:;
    if (!(x < 10))
      goto while_break;
    x++;
    if (x == 5 && nondet_int()) {
      x = 50;
      goto while_continue;
    }
  }
while_break:;
  __ESBMC_assert(x <= 10, "x <= 10");
  return 0;
}
