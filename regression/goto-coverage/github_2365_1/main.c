#include <assert.h>
#include <stdbool.h>

int b = 2;
int main()
{
  int a = nondet_int();
  // if (a == 1 && b == 2) { // |1|
  if ((a == 1 && b == 2) != 0)
  { // |2|
    assert(0 && "if");
  }
  else
    assert(0 && "else");
}