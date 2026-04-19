// Minimal reproducer: VLA with zero size from global variable.
// 'a' is a global int, initialized to 0.
// 'int b[a]' is a zero-length VLA, which is undefined behaviour (C11 §6.7.6.2p1).
int a;
int main(void) { int b[a]; return 0; }
