#include <stdio.h>

extern char __VERIFIER_nondet_char(void);
extern unsigned char __VERIFIER_nondet_uchar(void);
extern _Bool __VERIFIER_nondet_bool(void); 

int main(void) {
  char c = __VERIFIER_nondet_char();
  unsigned char uc = __VERIFIER_nondet_uchar();
  _Bool b = __VERIFIER_nondet_bool();

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