#include <assert.h>
#include <stdbool.h>

int b;
int main()
{
  int a = nondet_int();
  if (a == b && b == 2)
  { // |1|
    //      if ((int)(a == b && b == 2)) { // |2|
    //  if ((_Bool)(int)(a == b && b == 2)) { // |3|
    assert(0 && "if");
  }
  else
    assert(0 && "else");
}