#include <assert.h>
#include <limits.h>

int nondet_int();
unsigned int nondet_uint();

int main()
{
  int c_nondet = nondet_int();
  int d_nondet = nondet_int();
  int e_nondet = nondet_int();
  // (d - c) == (d - e) -> c == e
  assert(((d_nondet - c_nondet) == (d_nondet - e_nondet)) == (c_nondet == e_nondet));

  // Check cancellation near INT_MAX: (INT_MAX - c) == (INT_MAX - e)
  int c_max_1 = 1;
  int e_max_1 = 1;
  assert(((INT_MAX - c_max_1) == (INT_MAX - e_max_1)) == (c_max_1 == e_max_1));

  int c_max_2 = 1;
  int e_max_2 = 2; // Should fail
  assert(((INT_MAX - c_max_2) == (INT_MAX - e_max_2)) == (c_max_2 == e_max_2));

  // Check cancellation near INT_MIN (underflow/wraparound): (INT_MIN - c) == (INT_MIN - e)
  int c_min_1 = -1;
  int e_min_1 = -1;
  assert(((INT_MIN - c_min_1) == (INT_MIN - e_min_1)) == (c_min_1 == e_min_1));

  int c_min_2 = -1;
  int e_min_2 = -2; // Should fail
  assert(((INT_MIN - c_min_2) == (INT_MIN - e_min_2)) == (c_min_2 == e_min_2));

  // Minuend 'd' is INT_MAX, causing potential overflow with 'c' < 0
  int d_overflow = INT_MAX;
  int c_neg = -1; // INT_MAX - (-1) == INT_MAX + 1 -> Underflow to INT_MIN
  int e_neg = -1;
  assert(((d_overflow - c_neg) == (d_overflow - e_neg)) == (c_neg == e_neg));

  // Minuend 'd' is INT_MIN, causing potential underflow with 'c' > 0
  int d_underflow = INT_MIN;
  int c_pos = 1; // INT_MIN - 1 -> Wraparound to INT_MAX
  int e_pos = 1;
  assert(((d_underflow - c_pos) == (d_underflow - e_pos)) == (c_pos == e_pos));

  // Test with c = 0
  int d_zero_c = nondet_int();
  int e_val_c = 5;
  // (d - 0) == (d - 5) -> 0 == 5 (False)
  assert(((d_zero_c - 0) == (d_zero_c - e_val_c)) == (0 == e_val_c));

  // Test with e = 0
  int d_zero_e = nondet_int();
  int c_val_e = 5;
  // (d - 5) == (d - 0) -> 5 == 0 (False)
  assert(((d_zero_e - c_val_e) == (d_zero_e - 0)) == (c_val_e == 0));

  unsigned int c_u = nondet_uint();
  unsigned int d_u = nondet_uint();
  unsigned int e_u = nondet_uint();
  // (d - c) == (d - e) -> c == e (Should hold for unsigned integers as well)
  assert(((d_u - c_u) == (d_u - e_u)) == (c_u == e_u));

  // Test boundary case for unsigned (0 - 1 == UINT_MAX)
  unsigned int d_u_under = 0;
  unsigned int c_u_val = 1;
  unsigned int e_u_val = 1;
  assert(((d_u_under - c_u_val) == (d_u_under - e_u_val)) == (c_u_val == e_u_val));
  
  return 0;
}
