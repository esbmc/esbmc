#include <stdbool.h>
typedef struct {
	float x;
	bool z;
} myStruct;

int main(void) {
  myStruct testObj;
  testObj.x = 0.0;
  testObj.z = false;

  if (testObj.x > 10.0)
  {
    testObj.z = true;
  }

  assert(testObj.z); // fail
  return 0;
}
