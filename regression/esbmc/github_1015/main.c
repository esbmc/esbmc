int main() {
  // Defining sorts ...
  unsigned long mask = (unsigned long)-1 >> (sizeof(unsigned long) * 8 - 64);  
  unsigned long mask2 = (unsigned long)1 << (64 - 1);
  __ESBMC_assert(mask == 18446744073709551615UL, "Right shift should hold");
  __ESBMC_assert(mask2 == 9223372036854775808UL, "Left shift should hold");

  return 0;
}
