#include <cassert>
template <typename>
struct foo
{
  foo()
  {
    struct axy
    {
      foo *_M_guarded;
      int bar = 2;
    } __guard;
    __guard._M_guarded = 0;
    assert(__guard.bar == 22); // should be 2
  }
};
foo<char16_t> ss()
{
  foo<char16_t>{};
}
foo<char> s()
{
  foo<char32_t>{};
}
foo<char> __trans_tmp_1;
int main()
{
}
