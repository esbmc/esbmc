#include <assert.h>

extern unsigned long __VERIFIER_nondet_ulong(void);

/* A uint64_t bit-field wider than int ranks as no standard type; a compound
 * assignment with a 64-bit operand must still convert to a common width. */
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
  assert(pa.full_pa == (before & ~mask & 0xFFFFFFFFFFFFFul));
  pa.full_pa |= mask;
  assert(pa.full_pa == (((before & ~mask) | mask) & 0xFFFFFFFFFFFFFul));
  return 0;
}
