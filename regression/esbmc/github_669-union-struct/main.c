// main.i
typedef union { int x;
} a;
struct { int x; } b;
void func1() { a c = *(a *)&b; }
