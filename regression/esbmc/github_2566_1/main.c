#define FULP 1

void bug (float min) {
  __ESBMC_assume(min == 0x1.fffffep-105f);
  float modifier = (0x1.0p-23 * (1<<FULP));
  float ulpdiff = min * modifier;
  assert(ulpdiff == 0x1p-126f);    // Should be true
}

void bugBrokenOut (float min) {
  __ESBMC_assume(min == 0x1.fffffep-105f);
  float modifier = (0x1.0p-23 * (1<<FULP));
  double dulpdiff = (double)min * (double)modifier;  // OK
  float ulpdiff = (float)dulpdiff;  // Crash
  assert(ulpdiff == 0x1p-126f); // Should be true
}

void bugCasting (double d) {
  __ESBMC_assume(d == 0x1.fffffep-127);
  float f = (float) d;
  assert(f == 0x1p-126f); // Should be true
}

int main (void) {
  float f;
  bug(f);

  float g;
  bugBrokenOut(g);

  double d;
  bugCasting(d);

  return 1;
}
