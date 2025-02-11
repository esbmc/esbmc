#include <assert.h>
#include <stddef.h>
extern void __VERIFIER_assume(int);

int main(){
  __VERIFIER_assume(NULL != "a");
  char *str0 = (char *) 0xFFFFFFFFFFFFFFFF;
  char *str1 = "x";
  assert(str0 <= str1);
}
