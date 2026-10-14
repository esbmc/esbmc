#include <cassert>

// Negative of dtor_array_member_no_double_free: each element is destroyed
// exactly once, so asserting it is destroyed *twice* (the pre-fix double-free
// behaviour) must be violated -> VERIFICATION FAILED.

int destroyed[3];

struct R
{
  int id;
  ~R() { destroyed[id]++; }
};

struct Holder
{
  R rs[3];
  Holder()
  {
    rs[0].id = 0;
    rs[1].id = 1;
    rs[2].id = 2;
  }
};

int main()
{
  {
    Holder h;
  }
  assert(destroyed[0] == 2); // wrong on purpose: it is destroyed once
  return 0;
}
