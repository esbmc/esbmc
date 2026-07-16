#include <atomic>
#include <cassert>

int arr[4];

int main()
{
  std::atomic<int> a{5};
  assert(a.load() == 5);
  a.store(7);
  assert(a == 7);
  assert(a.exchange(9) == 7);
  assert(a.load() == 9);

  int expected = 9;
  assert(a.compare_exchange_strong(expected, 11));
  assert(a.load() == 11);
  expected = 99;
  assert(!a.compare_exchange_weak(expected, 13));
  assert(expected == 11);
  assert(a.load() == 11);

  assert(a.fetch_and(3) == 11);
  assert(a.load() == 3);
  assert(a.fetch_or(4) == 3);
  assert(a.load() == 7);
  assert(a.fetch_xor(1) == 7);
  assert(a.load() == 6);

  assert(++a == 7);
  assert(a++ == 7);
  assert(a.load() == 8);
  assert((a += 2) == 10);
  assert((a -= 4) == 6);
  assert(--a == 5);
  assert(a-- == 5);
  assert(a.load() == 4);

  std::atomic<int *> p{&arr[1]};
  assert(p.fetch_add(2) == &arr[1]);
  assert(p.load() == &arr[3]);
  assert(p.fetch_sub(1) == &arr[3]);
  assert(++p == &arr[3]);

  std::atomic_flag f = ATOMIC_FLAG_INIT;
  assert(!f.test_and_set());
  assert(f.test_and_set());
  f.clear();
  assert(!f.test());

  std::atomic_int ti{1};
  std::atomic_size_t ts{2};
  assert(ti.load() == 1 && ts.load() == 2);

  std::atomic_thread_fence(std::memory_order_acquire);
  return 0;
}
