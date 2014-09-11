#include <cassert>

template <class T>
class mycontainer
{
    T element;
public:
    mycontainer(T arg) { element=arg; }
    T increase() { return ++element; }
};

template <>
class mycontainer<float>
{
    float element;
public:
    mycontainer(float arg) { element=arg; }
    float test() { return element+2; }
};

template <>
class mycontainer<char>
{
    char element;
public:
    mycontainer(char arg) { element=arg; }
    char uppercase()
    {
        if (element >= 'a' && element <= 'z') {
            element += 'A' - 'a';
        }
        return element;
    }
};

int main ()
{
    mycontainer<int> myint(7);
    mycontainer<char> mychar('j');
    mycontainer<float> myfloat(1.0f);

    assert(myint.increase() == 8);
    assert(mychar.uppercase() == 'J');
    assert(myfloat.test() == 3.0f);
    return 0;
}
