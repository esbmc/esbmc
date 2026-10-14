// A function-local static with a dynamic initializer is initialized when
// control first passes its declaration ([stmt.dcl]/3), not before main.
#include <cassert>
struct S { int a; };
S gs = {1};
int get() { static S s(gs); return s.a; }
int main() { gs.a = 5; assert(get() == 5); gs.a = 9; assert(get() == 5); return 0; }
