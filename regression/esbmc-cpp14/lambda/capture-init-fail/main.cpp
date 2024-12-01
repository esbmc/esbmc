#include <cassert>

int main()
{
  int x = 42;
  int y = 1;

  auto lambda_with_capture_init_1 = [=, bar = 11]() {
    assert(x == 421);
    assert(y == 11);
    assert(bar == 111);
  };

  auto lambda_with_capture_init_2 = [x, y, bar = 11]() {
    assert(x == 421);
    assert(y == 11);
    assert(bar == 111);
  };

  auto lambda_with_capture_init_3 = [&, bar = 11]() {
    y = y + 1;
    assert(x == 421);
    assert(y == 21);
    assert(bar == 111);
  };

  auto lambda_with_capture_init_4 = [&x, &y, bar = 11]() {
    y = y + 1;
    assert(x == 421);
    assert(y == 31);
    assert(bar == 111);
  };

  int xx = 4;

  auto yy = [&r = xx, xx = xx + 1]() -> int {
    r += 2;
    return xx * xx;
  }(); // updates ::xx to 6 and initializes yy to 25.
  assert(yy == 251);
  assert(xx == 61);

  struct test
  {
    int first;
    int second;
  };

  struct test testStruct = {111, 444};
  auto yyy = [&r = testStruct, testStruct2 = test{}]() -> int {
    assert(r.first == 1111);
    assert(r.second == 4441);
    assert(testStruct2.first == 101);
    assert(testStruct2.second == 101);
    r.first++;
    r.second = 222;
    return r.second + r.first + testStruct2.first + testStruct2.second;
  }();
  assert(yyy == 3341);
  assert(testStruct.first == 1121);
  assert(testStruct.second == 2221);

  // Call the lambda
  lambda_with_capture_init_1();
  lambda_with_capture_init_2();
  lambda_with_capture_init_3();
  lambda_with_capture_init_4();

  return 0;
}
