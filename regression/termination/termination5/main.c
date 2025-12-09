// SPDX-FileCopyrightText: 2021 Y. Cyrus Liu <yliu195@stevens.edu>
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Date: 2021-06-21
 * Author: yliu195@stevens.edu
 */



#include <stdbool.h>

extern int __VERIFIER_nondet_int(void);

int v; // count the number of bits set in v
unsigned int c; // c accumulates the total bits set in v
int z, y;       // word value to compute the parity of

int main(){
  y = __VERIFIER_nondet_int();
  v= __VERIFIER_nondet_int();
  if (v>=0){
    for (c = 0; v; c++) {
      /* v &= v - 1; // clear the least significant bit set */
      v = v&(v - 1); // clear the least significant bit set
    }
    y = 1;
  } else {
    y = -1;
  }
  return z;
}
