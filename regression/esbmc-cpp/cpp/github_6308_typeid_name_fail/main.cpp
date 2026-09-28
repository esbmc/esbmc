// Negative companion to github_6308_typeid_name: two distinct types have
// distinct type_info, so asserting they are equal is violated.
#include <cassert>
#include <typeinfo>

struct A {};
struct B {};

int main()
{
  assert(typeid(A) == typeid(B)); // wrong on purpose: distinct types
  return 0;
}
