#include <assert.h>
#include <string.h>
#include <stdlib.h>
struct InnerStruct {
  char innerValue[10];
};

int main() {
  struct InnerStruct array[2];

  strcpy(array[0].innerValue, "20");
  strcpy(array[1].innerValue, "40");

  for (int i = 0; i < 2; i++) {
    int innerValue = atoi(array[i].innerValue);
    assert(innerValue == 40);
  }

  return 0;
}