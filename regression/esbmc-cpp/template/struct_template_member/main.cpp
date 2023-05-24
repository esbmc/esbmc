struct Foo {
    template<class T>
    T bar() {
       return T();
    }
};

int main(void) {
  Foo f;
  f.bar<int>();
  return 0;
}
