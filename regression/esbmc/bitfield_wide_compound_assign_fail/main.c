#include <assert.h>

extern unsigned long __VERIFIER_nondet_ulong(void);

union pa
{
  unsigned long raw;
  struct
  {
    unsigned long full_pa : 52;
    unsigned long hkid : 12;
  };
};

int main()
{
  union pa pa;
  pa.raw = __VERIFIER_nondet_ulong();
  unsigned long mask = __VERIFIER_nondet_ulong();
  unsigned long before = pa.full_pa;
  pa.full_pa &= ~mask;
  /* fails whenever mask clears a set bit of full_pa */
  assert(pa.full_pa == before);
  return 0;
}
