/*
$clang -fsanitize=vla-bound -g main.c -o test
$./test
main.c:3:10: runtime error: variable length array bound evaluates to non-positive value 0
SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior main.c:3:10 
*/

int a;
int main() {
  char c[a];
  (void)sizeof(c);
}
