#include <stdio.h>
#include <string.h>
const int t[2] = {1, 2};
const int c = 3;
void __VERIFIER_nondet_memory(void *, size_t);
int main()
{
  int s[2] = {7, 8};
  __VERIFIER_nondet_memory((void *)t, sizeof t);
  return 0;
}
