#include <stdio.h>

extern int __VERIFIER_nondet_int(void);

static int compute(int a, int b) {
  if (a > 0) {         
    if (b == 0) {     
      return 10;
    } else {
      return 11;
    }
  } else {            
    if (b < 0) {       
      return 20;
    } else {
      return 21;
    }
  }
}

int main(void) {
  int a = __VERIFIER_nondet_int();  
  int b = __VERIFIER_nondet_int();  
  int r = compute(a, b);
  return 0;
}
