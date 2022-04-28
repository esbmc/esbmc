typedef struct {
	float x;
} myStruct;

int main(void) {
  myStruct testObj;
  testObj.x = 200.0;
  float z = 0.0;

  if (testObj.x > 10.0f)
  {
    z = testObj.x;
  }

  assert(z == 0.0); // fail
  return 0;
}
