// esbmc/esbmc#1866: non-first base subobject under multiple inheritance.
// KNOWNBUG until the base-subobject rework (#3894) lands; correct result is
// VERIFICATION SUCCESSFUL (verified against clang++/g++). See
// docs/design/cpp-multiple-inheritance-subobjects.md
#include <cassert>
struct B { int e = 22; int getE(){return e;} };
struct A { int b = 111; int getB(){return b;} };
struct f : B, A {};
int main(){ f i=f(); assert(i.getE()==22); assert(i.getB()==111); return 0; }
