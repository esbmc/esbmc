#include <cassert>

// Aggregate template (C++20 P1816 implicit aggregate guides).
template <typename T>
struct Box
{
  T t;
};

// Template with user-provided constructor.
template <typename T>
struct Wrap
{
  T t;
  Wrap(T x) : t(x)
  {
  }
};

// Template with explicit deduction guide.
template <typename T>
struct Holder
{
  T value;
};
template <typename T>
Holder(T) -> Holder<T>;

int main()
{
  // Aggregate CTAD.
  Box b{42};
  assert(b.t == 42);

  // Constructor-based CTAD.
  Wrap w{99};
  assert(w.t == 99);

  // Explicit guide.
  Holder h{7};
  assert(h.value == 7);

  return 0;
}
