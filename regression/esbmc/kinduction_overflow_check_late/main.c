// The overflow needs eleven iterations. The inductive step runs from a
// havoced i, finds i == 10 and fails, so no k below 11 can claim success;
// the base case reaches the overflow at k = 11.
extern int __VERIFIER_nondet_int(void);

int main(void) {
  int i = 0;
  while (__VERIFIER_nondet_int()) {
    if (i < 10)
      i = i + 1;
    else
      i = i + 2147483647;
  }
  return 0;
}
