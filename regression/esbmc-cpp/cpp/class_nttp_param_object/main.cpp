// A C++20 class-type template argument: clang cannot generate a USR for the
// specialisation or for the template parameter object it names.
#include <cassert>
struct S { int x; };
template <S s> int g() { return s.x; }
int main() { assert(g<S{1}>() == 1); assert(g<S{2}>() == 2); }
