inline __attribute__((__always_inline__)) int
__inline_signbitl(long double __x) {
  union {
    long double __ld;
    struct {
      unsigned long long __m;
      unsigned short __sexp;
    } __p;
  } __u;
  __u.__ld = __x;
  return (int)(__u.__p.__sexp >> 15);
}

int main(int argc, char **argv) {
  float f = -0x1p-129f;
  float g = 0x1p-129f;
  float target = 0x0;

  float result = f + g;
  assert(__inline_signbitl(result) == __inline_signbitl(target));
  return 0;
}
