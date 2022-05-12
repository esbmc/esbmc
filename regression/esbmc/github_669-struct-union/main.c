// main.i
typedef struct { int x;
} a;
union { int x; } b;
void func1() { a c = *(a *)&b; }
