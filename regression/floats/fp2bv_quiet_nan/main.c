#include <assert.h>
#include <string.h>

unsigned nondet_uint(void);
unsigned long long nondet_ulonglong(void);

int main(void)
{
  unsigned zb = nondet_uint();
  __ESBMC_assume(zb == 0);
  float z;
  memcpy(&z, &zb, sizeof z);
  float f = z / z;
  unsigned fbits;
  memcpy(&fbits, &f, sizeof fbits);
  assert(fbits == 0x7fc00000u);

  unsigned long long zdb = nondet_ulonglong();
  __ESBMC_assume(zdb == 0);
  double zd;
  memcpy(&zd, &zdb, sizeof zd);
  double d = zd / zd;
  unsigned long long dbits;
  memcpy(&dbits, &d, sizeof dbits);
  assert(dbits == 0x7ff8000000000000ull);
  return 0;
}
