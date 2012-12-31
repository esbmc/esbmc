#include <stack>
#include <cassert>
int main()
{
    std::stack<int> s;

    for (int i = 0; i < 10; ++i) {
        s.push(i);
    }

    for (int i = 9; i >= 0; --i) {
        assert(s.top() == i);
        s.pop();
    }

    return 0;
}
