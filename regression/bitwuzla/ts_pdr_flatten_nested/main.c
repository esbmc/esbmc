int main() {
  unsigned char i = 0, total = 0;
  while (i < 4) {
    unsigned char j = 0;
    while (j < 3) {
      j++;
      total++;
    }
    i++;
  }
  __ESBMC_assert(total == 12, "nested");
  return 0;
}
