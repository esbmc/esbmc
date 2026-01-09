/* Basic function contract test case
 * Tests __ESBMC_requires, __ESBMC_ensures clauses
 */
 #include <assert.h>

 int increment(int x)
{
   __ESBMC_requires(x > 0);
   __ESBMC_ensures(__ESBMC_return_value > x);
   return x + 1;
 }
 
 int main()
{
   int a = 5;
   int result = increment(a);
   
   // The contract ensures result > a, so this should always hold
   assert(result > a);
   
   return 0;
 }