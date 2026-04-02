// Regression for negative VLA dimension (complement to github_1533_2).
// 'a' is a global int, initialized to -1.
// 'int b[a]' is a negative-length VLA, which is undefined behaviour (C11 §6.7.6.2p1).
int a = -1;
int main(void) { int b[a]; return 0; }
