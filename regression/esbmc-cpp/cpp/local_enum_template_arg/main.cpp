// A class template specialised on same-named local enums of two functions
// gives two distinct specialisations with their own layouts.
#include <cassert>
#include <cstring>
template <class T> struct Box { T t; int w; };
int f() { enum class E : char { A = 1 }; Box<E> b; std::memset(&b, 0, sizeof b); b.w = 5; return *((int*)&b + 1); }
int g() { enum class E : long { A = 1 }; Box<E> b; std::memset(&b, 0, sizeof b); b.w = 5; return *((int*)&b + 2); }
int main() { assert(f() == 5); assert(g() == 5); return 0; }
