#include <deque>
#include <queue>
#include <set>
#include <stack>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

int main()
{
  {
    std::deque<int> d;
    d.push_back(1);
    std::set<int> s;
    s.insert(2);
    std::multiset<int> ms;
    ms.insert(2);
    std::stack<int> st;
    st.push(3);
    std::queue<int> q;
    q.push(4);
    std::unordered_map<int, int> m;
    m[4] = 5;
    std::unordered_set<int> us;
    us.insert(6);
    std::tuple<int, int> t(1, 2);
  }
  return 0;
}
