#include <c_interface.h>
#include <cstdio>

int main() {
  printf("%s", get_git_version_tag());
  return 0;
}