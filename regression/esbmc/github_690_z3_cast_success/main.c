typedef struct {
	float x;
} myStruct;

int main(void) {
  myStruct testObj;
  testObj.x = 200.0;
  double y = 10.0;
  float z = 0.0;

  if (testObj.x > y)
  {
    z = testObj.x;
  }
  else
  {
    z = y;
  }

  assert(z == testObj.x); // success
  return 0;
}
