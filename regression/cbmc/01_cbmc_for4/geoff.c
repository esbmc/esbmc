#include <string.h>
#include <stdio.h>

int __CRTMC_lines_len = 7;
char * __CRTMC_lines[] = {"#  nq @$!i   QTg#l     ", "1 10 ", "37 42 52 14 54 11 44 18 29 16 ", "2 2", "2 1 1 1 ", "2 2", "1.0 2.0 3 4"};

#define MAXLINE 32
#define COMMENT '#'

// Check that function calls in the iteration statement of for loops, that also
// contain a side-effect, don't get discarded. Do this by testing the
// reachability of the assertion in acall.

void acall(char *a, char *b, int c) {
  assert(0);
}

int main() {
	char line[MAXLINE];
	int __CRTMC_iter;
for(__CRTMC_iter = 0; __CRTMC_iter < __CRTMC_lines_len; acall(line, __CRTMC_lines[__CRTMC_iter++], MAXLINE)) {

}

return 0;
}
