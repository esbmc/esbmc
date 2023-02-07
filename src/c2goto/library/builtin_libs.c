int __ESBMC_sync_fetch_and_add(int *ptr, int value)
{
  __ESBMC_atomic_begin();
  int initial = *ptr;
  *ptr += value;
  __ESBMC_atomic_end();
  return initial;
}

#ifdef __CHERI__

#include <stdint.h>
// #include <stddef.h>
// #include <cheri.h>

__SIZE_TYPE__ __esbmc_cheri_length_get(void *__capability cap)
{
  union {
    void *__capability cap;
    struct {
      uint64_t perms : 16;
      uint64_t : 3;
      uint64_t otype : 18;
      uint64_t I_E : 1;
      uint64_t T_11_3 : 9;
      uint64_t T_E : 3;
      uint64_t B_13_3 : 11;
      uint64_t B_E : 3;
      uintptr_t a;
    };
  } u = { cap };
  uint64_t E;
  uint64_t T = u.T_11_3 << 3;
  uint64_t B = u.B_13_3 << 3;
  int L_carry_out;
  int L_msb;
  if (u.I_E) {
    E = u.T_E << 3 | u.B_E;
    L_carry_out = (T & 0x7f8) < (B & 0x7f8);
    L_msb = 1;
  } else {
    E = 0;
    T |= u.T_E;
    B |= u.B_E;
    L_carry_out = (T & 0x7ff) < (B & 0x7ff);
    L_msb = 0;
  }
  T |= ((B >> 12) + L_carry_out + L_msb) << 12;
  uint64_t a_top = u.a & -(1 << (E + 14));
  uint64_t a_mid = u.a & (-(1 << 13) << E);
  uint64_t a_low = u.a & ((1 << (E - 1)) - 1);
  int c_t, c_b;
  uint64_t A3 = (u.a >> 11) & 0x7;
  uint64_t B3 = (B >> 11) & 0x7;
  uint64_t T3 = (T >> 11) & 0x7;
  uint64_t R = B3 - 1;
  switch ((A3 < R) << 1 | (T3 < R)) {
  case 0: c_t = 0; break;
  case 1: c_t = 1; break;
  case 2: c_t = -1; break;
  case 3: c_t = 0; break;
  }
  switch ((A3 < R) << 1 | (B3 < R)) {
  case 0: c_b = 0; break;
  case 1: c_b = 1; break;
  case 2: c_b = -1; break;
  case 3: c_b = 0; break;
  }
  uint64_t t = (a_top + c_t) << (E + 14) | T;
  uint64_t b = (a_top + c_b) << (E + 14) | B;
  return t - b;
}

#endif
