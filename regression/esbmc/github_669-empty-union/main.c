// main.i
typedef union {
} a;
int b;
void func1() { a c = *(a *)&b; }
