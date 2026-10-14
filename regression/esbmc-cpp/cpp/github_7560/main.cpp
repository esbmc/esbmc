// A member reached through an anonymous union or struct is an
// IndirectFieldDecl, a fourth CXXCtorInitializer kind the dispatch chain never
// tested, so the constructor aborted with SIGABRT and no verdict. The member
// also sits inside the anonymous field, not beside it (#7560).
#include <cassert>

struct WithUnion
{
  union
  {
    int a;
    float b;
  };
  int tail;
  WithUnion() : a(1), tail(2)
  {
  }
};

struct WithStruct
{
  struct
  {
    int x;
    int y;
  };
  WithStruct() : x(3), y(4)
  {
  }
};

int main()
{
  WithUnion u;
  assert(u.a == 1);
  assert(u.tail == 2);

  WithStruct s;
  assert(s.x == 3);
  assert(s.y == 4);
  return 0;
}
