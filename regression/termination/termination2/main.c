// SPDX-FileCopyrightText: 2021 Y. Cyrus Liu <yliu195@stevens.edu>
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Date: 2021-06-21
 * Author: yliu195@stevens.edu
 */

extern int __VERIFIER_nondet_int(void);

int main (){
    int x, y, res;
    x = __VERIFIER_nondet_int();
    y = __VERIFIER_nondet_int();
    while (x>=y && y > 0){
        x = (x-1)&y;
    }
    return 0;
}
