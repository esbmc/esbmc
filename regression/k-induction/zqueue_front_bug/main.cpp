// queue::front
#include <cassert>

namespace std
{
  #define QUEUE_CAPACITY 1000

  template<class T> class queue
  {
    T buf[QUEUE_CAPACITY];
    int _size;
    int head;
    int tail;
	
    public:

    queue():_size(0), head(0), tail(0){}

    void push ( const T& t )
    {
      assert(0 <= _size);
      assert(_size < QUEUE_CAPACITY);
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
      assert(!empty());
      return buf[head];
    }

    int size() const
    {
      assert(0 <= _size && _size <= QUEUE_CAPACITY);
      return _size;
    }

    void pop()
    {
      assert(size()>0);
      _size--;
      if (head == QUEUE_CAPACITY) 
        head = 1;
      else 
        head++;
    }

    T& back ()
    {     
      assert(!empty());
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
  for(i = N; i >= 0; --i)
    myqueue.push(i);

  assert(myqueue.front() == N);

  return 0;
}
