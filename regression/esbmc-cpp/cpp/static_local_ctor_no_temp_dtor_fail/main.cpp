// A function-local static with a dynamic initializer is initialized when
// control first passes its declaration ([stmt.dcl]/3), not before main.
#include <cassert>
int dtors=0;
struct T { int v; T(int x):v(x){} ~T(){++dtors;} };
int get(int x){ static T t(x); return t.v; }
int main(){ get(1); assert(dtors==1); return 0; }
