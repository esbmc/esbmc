unsigned nondet_uint(void);
int nondet_int(void);
int main(void) {
  unsigned a = nondet_uint(), b = nondet_uint();
  int x = nondet_int(), y = nondet_int();
  unsigned m = a * b;
  unsigned um = a % b;
  int sm = x % y;
  unsigned bn = ~a;
  unsigned ba = a & b, bo = a | b, bx = a ^ b;
  unsigned sl = a << 1, sr = a >> 1;
  int as = x >> 1;
  unsigned t = a ? a : b;
  _Bool c = a < b;
  int lt = x < y, le = x <= y, gt = x > y, ge = x >= y;
  int ult = a < b, ule = a <= b, ugt = a > b, uge = a >= b;
  int eq = a == b, ne = a != b;
  int nt = !x, an = x && y, orr = x || y;
  long wide = x;
  unsigned long uw = a;
  unsigned char nar = a;
  unsigned same = x;
  _Bool tb = 1;
  __ESBMC_assert(a != b || 1, "ne");
  return 0;
}
