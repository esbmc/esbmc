#include <assert.h>
#include <stdbool.h>
typedef struct {
  bool (*a)();
  void (*act)();
} j;
j f;
bool b() { return 1; }
void d() { assert(0); }
bool c() { return 1; }
void e() {};
int main() {
  j actions[2];
  actions[0] = (j){b, d};
  actions[1] = (j){c, e};
  j *g = actions;
  {
    j *actions = g;
    for (int i = 0; i < 2; i++)
      if (actions[i].a())
        f = g[i];
  }
  f.act();
  return 0;
}
