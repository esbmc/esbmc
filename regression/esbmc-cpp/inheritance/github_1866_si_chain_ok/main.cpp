// Baseline: 3-level single-inheritance chain already verifies correctly.
// Guards against the #3894 base-subobject rework regressing single inheritance.
#include <cassert>
struct A { int a = 1; };
struct B : A { int b = 2; };
struct C : B { int c = 3; };
int main(){ C i=C(); assert(i.a==1); assert(i.b==2); assert(i.c==3); return 0; }
