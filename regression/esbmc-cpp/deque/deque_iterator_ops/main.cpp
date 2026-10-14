#include <cassert>
#include <deque>

int main()
{
  std::deque<int> d;
  d.push_back(10);
  d.push_back(20);
  d.push_back(30);

  std::deque<int>::iterator it = d.begin();
  it += 2;
  assert(*it == 30);

  --it;
  assert(*it == 20);

  it -= 1;
  assert(*it == 10);

  std::deque<int>::iterator post = it++;
  assert(*post == 10);
  assert(*it == 20);

  return 0;
}
