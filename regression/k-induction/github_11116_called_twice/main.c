int g, x;
void f() {
  for (int i = 0; i < 6; i++) {
    __ESBMC_assert(g == 0 || x != 0, "P");
    x = 1;
  }
}
int main() { x = 0; g = 0; f(); x = 0; g = 1; f(); }
