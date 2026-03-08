extern "C" {
  char __VERIFIER_nondet_char(void);
  unsigned char __VERIFIER_nondet_uchar(void);
  bool __VERIFIER_nondet_bool(void);
}

int main() {
  char c = __VERIFIER_nondet_char();
  unsigned char uc = __VERIFIER_nondet_uchar();
  bool b = __VERIFIER_nondet_bool();

  if (b) {
    if (c >= 'A') {
      return 1;
    } else {
      return 2;
    }
  } else {
    if (uc > 128) {
      return 3;
    } else {
      return 0;
    }
  }
}
