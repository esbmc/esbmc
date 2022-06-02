// main.i
typedef union { float x;
} a;
int b;
void func1() { a c = *(a *)&b; }
