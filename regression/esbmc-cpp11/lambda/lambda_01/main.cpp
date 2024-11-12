#include <cassert>

int main()
{
    int x = 42;
    int y = 1;

    auto lambda_with_capture_1 = [=]()
    {
        assert(x == 42);
        assert(y == 1);
    };

    auto lambda_with_capture_2 = [x, y]()
    {
        assert(x == 42);
        assert(y == 1);
    };

    auto lambda_with_capture_3 = [&]()
    {
        y++;
        assert(x == 42);
        assert(y == 2);
    };

    auto lambda_with_capture_4 = [&x, &y]()
    {
        y++;
        assert(x == 42);
        assert(y == 3);
    };

    // Call the lambda
    lambda_with_capture_1();
    lambda_with_capture_2();
    lambda_with_capture_3();
    lambda_with_capture_4();

    return 0;
}
