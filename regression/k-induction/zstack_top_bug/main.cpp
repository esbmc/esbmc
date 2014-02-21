// stack::top
#include <cassert>

namespace std
{
  #define STACK_CAPACITY 1000
  template<class T> class stack
  { 
    T buf[STACK_CAPACITY];
    int _top=0;

    public:
    stack():_top(0){}

    void inc_top ( )
    {
      _top++;
    }

    void dec_top ( )
    {
      _top--;
    }

    int get_top ( )
    {
      return _top;
    }

    void push ( const T& t )
    {
      __ESBMC_assert(0 <= _top, "invalid top");
      __ESBMC_assert(_top < STACK_CAPACITY, "stack overflow");
      buf[get_top()] = t;
      inc_top();
    }

    bool empty ( ) const
    {
      return (_top==0) ? true : false; 
    }

    int size() const
    {
      __ESBMC_assert(0 <= _top && _top <= STACK_CAPACITY, "invalid top");
      return _top;
    }

    void pop ( )
    {
      __ESBMC_assert(_top>0, "stack underflow");
      dec_top();
    }

    T& top ( )
    {
      assert(!empty());
      return buf[_top-1];
    }

    const T& top ( ) const
    {
      assert(!empty());
      return (const T)buf[_top-1];
    }

  };
}

using namespace std;

int nondet_int();
int N = nondet_int();

int main ()
{
  stack<int> mystack;

  __ESBMC_assume(N>0);

  for(int i=0; i <= N; ++i)
    mystack.push(i);

  assert(mystack.top() == N);
  return 0;
}
