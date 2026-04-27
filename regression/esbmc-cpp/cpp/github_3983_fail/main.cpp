#include <stack>

int main()
{
  std::stack<int> st;
  st.push(10);
  st.pop();
  int x = st.top(); // stack underflow: __ESBMC_assert fires inside top()
  (void)x;
  return 0;
}
