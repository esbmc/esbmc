#include <cassert>

void f() noexcept {}
void g() {}

template <typename T>
bool tf_is_noexcept()
{
  // noexcept(f()) is value-dependent during template parsing of this body
  // (T is unused but the body is still parsed in dependent context). Once
  // the template is instantiated below, Clang resolves it to a definite
  // value and the IR sees true/false.
  return noexcept(f());
}

template <typename T>
bool tg_is_noexcept()
{
  return noexcept(g());
}

int main()
{
  assert(tf_is_noexcept<int>() == true);
  assert(tg_is_noexcept<int>() == false);
  return 0;
}
