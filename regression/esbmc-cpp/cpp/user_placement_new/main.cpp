// A new-expression with placement arguments calls the operator new they
// select ([expr.new]/16), not a fresh built-in allocation.
#include <cassert>
#include <cstddef>

struct Pool
{
  alignas(8) unsigned char buf[64];
  std::size_t used = 0;
};

void *operator new(std::size_t n, Pool &p)
{
  void *r = p.buf + p.used;
  p.used += n;
  return r;
}

void *operator new[](std::size_t n, Pool &p)
{
  void *r = p.buf + p.used;
  p.used += n;
  return r;
}

struct S
{
  int v;
  S(int x) : v(x)
  {
  }
  static void *operator new(std::size_t n, Pool &p, int skip)
  {
    p.used += skip;
    return ::operator new(n, p);
  }
};

int calls = 0;
int skip()
{
  ++calls;
  return 4;
}

int main()
{
  Pool pool;
  int *a = new (pool) int(7);
  int *b = new (pool) int[3]{1, 2, 3};
  assert((unsigned char *)a == pool.buf);
  assert((unsigned char *)b == pool.buf + sizeof(int));
  assert(*a == 7 && b[2] == 3);

  S *s = new (pool, skip()) S(9);
  assert(calls == 1);
  assert((unsigned char *)s == pool.buf + 5 * sizeof(int));
  assert(s->v == 9);
  assert(pool.used == 6 * sizeof(int));
  return 0;
}
