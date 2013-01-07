#include <cstddef>
#include <vector>

class A {
  public:
    A();
};

template<class T>
class list {
public:

	struct node {
		T data;
		node* prev;
		node* next;
		node(T t, node* p, node* n) :
				data(t), prev(p), next(n) {
		}
	};
	int _size;

	class iterator {
	public:
		node* it;
        int it_size;

		bool operator ==(const iterator& x) const{
			if(this->it->data != x.it->data) //ERRO
				return false; //ERRO
			return true;
		}
	};
};

int main ()
{
  list<A> mylist;

  return 0;
}
