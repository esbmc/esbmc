#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

bool func() { return notdet_bool(); }

int main()
{
  bool a = nondet_bool();
  bool b =  true;
  auto c = nondet_bool();
  if (b && c|| a && b) 
  {
    assert(1);
  }
  else
    assert(0);
}
