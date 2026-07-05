// Companion to github_5142_int_conversion: the same implicit int->pointer
// conversion (which only parses with -Wno-int-conversion) must still be
// verified, not rubber-stamped. Here the round-trip value is 0xdead, so the
// assertion that it equals 0 is violated and ESBMC must report FAILED.
int main(void)
{
  void *p = 0xdeadUL; // implicit int -> pointer
  unsigned long v = (unsigned long)p;
  __ESBMC_assert(v == 0, "v is 0xdead, not 0 -- must fail");
  return 0;
}
