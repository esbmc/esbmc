// queue::push/pop
#include <cassert>

namespace std
{
  #define QUEUE_CAPACITY 1000

  template<class T> class queue
  {
    public:

    T buf[QUEUE_CAPACITY];
    int _size;
    int head;
    int tail;
	
    queue():_size(0), head(0), tail(0){}

    void push ( const T& t )
    {
      assert(0 <= _size);
      __ESBMC_assert(_size < QUEUE_CAPACITY, "queue overflow");
      buf[tail] = t;
      _size++;
     if (tail == QUEUE_CAPACITY)
       tail = 1;
     else
       tail++;
    }

    bool empty ( ) const
    {
      if (head == tail) 
        return true;
      else 
	return false;
    }

    T& front ( )
    {
      assert(!(head == tail));
      return buf[head];
    }

    int size() const
    {
      assert(0 <= _size && _size <= QUEUE_CAPACITY);
      return _size;
    }

    void pop()
    {
      __ESBMC_assert(_size>0, "queue underflow");
      _size--;
      if (head == QUEUE_CAPACITY) 
        head = 1;
      else 
        head++;
    }

    T& back ()
    {     
      assert(!(head == tail));
      return buf[tail-1];
    }    
  };
}
using namespace std;

int nondet_int();
int N = nondet_int();

int main ()
{
  queue<int> myqueue;
  int myint;

  __ESBMC_assume(N>0);

  int i;
  for(i = 0; i < N; i++)
    myqueue.push(i);
   
  while (!myqueue._size)
  {
    assert(myqueue.front() != N-i--);
    myqueue.pop();
  }

  return 0;
}
