/*
 * This program aims to allow ESBMC to take advantage of interactive verification 
 * with ESBMC using ACSL (ANSI/ISO C Specification Language) annotations.
 * The example features a simple loop with ACSL loop invariant annotations
 * to verify correctness properties through formal verification. 
 * Should pass
 */

 #include <assert.h>

 int main() {
     int i = 0;
     int sum = 0;
 
     /*@ loop invariant i >= 0 && i <= 5000000 && sum == i * 10;
     @*/
     
     //first check the base case

    //should havoc the variables: change the invovlved variables to non deterministic values
    //use assumption to set i >= 0 && i <= 5000000 && sum == i * 10;

    __ESBMC_loop_invariant(i >= 0 && i <= 5000000 && sum == i * 10);
     while (i < 5000000)//replace the condition to i <5000000
     {   
        sum += 10;
        i++;
         // check the loop invariant after the update
         // assert the i >= 0 && i <= 5000000 && sum == i * 10;
         // if the assertion fails, then the loop invariant is not maintained
         // and should report the invariant violation


         // should use assume(FALSE) to terminate the loop
     }
     // no matter if the loop invariant is maintained, we should use the invariant to check the post condition
     // assume i >= 0 && i <= 5000000 && sum == i * 10;
     assert(sum == 50000000);     // 5000000 * 10 //should be true
     return 0;
 }