#include <stdio.h>
#include <string.h>
const int t[2] = {1, 2};
const int c = 3;
void __VERIFIER_nondet_memory(void *, size_t);
int main()
{
  int s[2] = {7, 8};
  sscanf("5", "%d", (int *)&c);
  return 0;
}
