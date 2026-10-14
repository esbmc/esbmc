// valarray::max, min, shift, cshift and both apply overloads were DECLARED but
// never DEFINED, so each call returned a nondeterministic value: on {5,2,9,2},
// `a.max()` satisfied neither `== 9` nor `!= 9`. Silent wrong answers rather
// than a diagnostic.
//
// Verified against clang++ -std=c++17 -fsanitize=address,undefined: exits 0.
#include <valarray>
#include <cassert>

int twice(int x)
{
  return 2 * x;
}

int main()
{
  std::valarray<int> a(4);
  a[0] = 5;
  a[1] = 2;
  a[2] = 9;
  a[3] = 2;

  assert(a.max() == 9);
  assert(a.min() == 2);
  assert(a.sum() == 18);
  assert(a.size() == 4);

  // shift: positive moves toward the front, vacated slots are value-initialised
  std::valarray<int> s = a.shift(1);
  assert(s[0] == 2 && s[1] == 9 && s[2] == 2 && s[3] == 0);

  std::valarray<int> s2 = a.shift(-1);
  assert(s2[0] == 0 && s2[1] == 5 && s2[2] == 2 && s2[3] == 9);

  // cshift: rotates, nothing is lost
  std::valarray<int> c = a.cshift(1);
  assert(c[0] == 2 && c[1] == 9 && c[2] == 2 && c[3] == 5);

  std::valarray<int> c2 = a.cshift(-1);
  assert(c2[0] == 2 && c2[1] == 5 && c2[2] == 2 && c2[3] == 9);

  // a full rotation is the identity
  std::valarray<int> c3 = a.cshift(4);
  assert(c3[0] == 5 && c3[1] == 2 && c3[2] == 9 && c3[3] == 2);

  std::valarray<int> ap = a.apply(twice);
  assert(ap[0] == 10 && ap[2] == 18);

  return 0;
}
