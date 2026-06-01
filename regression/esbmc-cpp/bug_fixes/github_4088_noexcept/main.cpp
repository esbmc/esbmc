#include <cassert>

struct NX
{
  void m() noexcept {}
};

struct TH
{
  void m() {}
};

// Exercises the CXXNoexceptExpr conversion arm in the C++ frontend across
// two instantiations of the same template. Each instantiation produces a
// non-dependent noexcept(t.m()) that resolves to true / false, covering the
// getValue() branch added in #4088.
template <typename T>
bool is_m_noexcept()
{
  T t;
  return noexcept(t.m());
}

int main()
{
  assert(is_m_noexcept<NX>() == true);
  assert(is_m_noexcept<TH>() == false);
  return 0;
}
