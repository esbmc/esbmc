template <typename T>
struct intrusive_ptr
{
  T *ptr = nullptr;

  intrusive_ptr() = default;
  explicit intrusive_ptr(T *p) : ptr(p)
  {
  }

  // When asking for this template, clang gives us
  // instrusive_ptr<Allocator>. Processing requires
  // processing Allocator, which loops back to Base.
  // ESBMC was set up to assume that Base was processed
  // after this "looping back", but it isn't!
  // The logic bailed out early because Base is already being processed
  // earlier in the call stack.   
  template <typename Y>
  intrusive_ptr(intrusive_ptr<Y> const &r) : ptr(r.ptr)
  {
  }
};

struct Base
{
  virtual ~Base()
  {
  }

  void force_processing_of_intrustive_ptr_Base()
  {
    intrusive_ptr<Base> b;
  }

  virtual void execute_deferred()
  {
  }
};

struct Allocator : Base
{
};


void somewhere_else_in_the_program()
{
  Allocator a;
  intrusive_ptr<Allocator> pa(&a);
  intrusive_ptr<Base> pb = pa; // upcast -- instantiates the Y=Allocator ctor
  (void)pb;
}

int main()
{
}
