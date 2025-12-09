#include <cassert>

int main()
{
  int x = 42;
  int y = 1;

  auto lambda_with_capture_1 = [=]() {
    assert(x == 1); //fail
    assert(y == 1);
  };

  auto lambda_with_capture_2 = [x, y]() {
    assert(x == 42);
    assert(y == 0); //fail
  };

  auto lambda_with_capture_3 = [&]() {
    y = y + 1;
    assert(x == 42);
    assert(y == 2);
  };

  auto lambda_with_capture_4 = [&x, &y]() {
    y = y + 1;
    assert(x == 42);
    assert(y == 3);
  };

  // Call the lambda
  // The first two lambdas capture by value, so we should be able to freely modify x and y
  x = 11;
  y = 11;
  lambda_with_capture_1();
  lambda_with_capture_2();
  // The last two lambdas capture by reference, so change x and y back
  x = 42;
  y = 1;
  lambda_with_capture_3();
  lambda_with_capture_4();

  return 0;
}
