// priority_queue::size
#include <cassert>
#include <cstdlib>

namespace std
{
  #define QUEUE_CAPACITY 1000

//  typedef int size_t;

  template < class T >
  class priority_queue
	{
	    T buf[QUEUE_CAPACITY];
	    int _size=0;
	    T* head;
	    T* tail;
	public:
	  priority_queue ( ):_size(0), head(0), tail(0){}
	  priority_queue ( T& x, T& y ):_size(0), head(0), tail(0){}
	  priority_queue ( T* x, T* y ):_size(0), head(0), tail(0){
		  for (;x!=y;x++)
			  this->push(*x);
	  }

	  void push ( T& x ){
		  int i = 0;
		  T tmp = x, y;
		  while((x < this->buf[i]) && (i != this->_size)){
			  i++;
		  }
		  if (i == 0)
			  this->head = &x;
		  if (i == this->_size)
			  this->tail = &x;
			  for (;i< this->_size;i++){
				  y = tmp;
				  tmp = this->buf[i];
				  this->buf[i] = y;
			  }
			  this->buf[i] = tmp;
			  this->_size++;
	  }
	  bool empty ( ) const{
		  if (this->_size == 0)
			  return true;
		  return false;
	  }
	  size_t size ( ) const{
		  return this->_size;
	  }
	  T top ( ) const{
		  //__ESBMC_assert(!(this->empty()), "the queue is empty");
		  return *this->head;
	  }
	  void pop ( ){
		  //__ESBMC_assert(!(this->empty()), "the queue is empty");
		  int i;
		  if (this->_size == 1){
			  this->head = this->tail = NULL;
			  this->_size--;
			  return;
		  }
		  for (i = 0; i < this->_size; i++){
			  this->buf[i] = this->buf[i+1];
		  }
		  this->head = &this->buf[0];
		  this->_size--;
		  this->tail = &this->buf[this->_size-1];
	  }
	};
}
using namespace std;

//int nondet_int();
//int N = nondet_int();

int main ()
{
  int N=10;
//  __ESBMC_assume(N>0);

  priority_queue<int> myints;

  for (int i=0; i<=N; i++) 
    myints.push(i);

  assert(myints.size() == N+1);
  myints.pop();
  assert(myints.size() == N);
  return 0;
}
