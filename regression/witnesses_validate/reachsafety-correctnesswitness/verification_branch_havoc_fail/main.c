extern int __VERIFIER_nondet_int(void);
int main() {
  int i = 0, s = __VERIFIER_nondet_int();
  while (i < 4) {
    s = s + 1;
    i++;
  }
  return s;
}
