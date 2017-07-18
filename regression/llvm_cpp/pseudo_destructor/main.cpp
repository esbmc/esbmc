template<typename T>
void destroy(T* ptr) {
  ptr->~T();
}

int main(void)
{
  int x;
  destroy(&x);
  return 0;
}
