// Regression for the IREP2-native rw_set data-race analysis (#4715, B5
// Phase 2.1/2.2 -- #4718/#4719). Exercises three of the four expr2tc
// paths the migration reshaped from a single race-positive program:
//   - array-index access  (is_index2t)
//   - struct member       (is_member2t)
//   - if-guarded write    (is_if2t with the true/false guard split)
// The fourth path (call-site read via is_code_function_call2t) is
// covered by the companion unit test in unit/goto-programs/rw_set.test.cpp.
//
// Two threads concurrently write the same locations on each path, so
// --data-races-check must fire. Pins the FAIL verdict to the IREP2 path.
// Modelled after regression/esbmc-unix/00_race01 -- no joins, so the
// interleaving search converges quickly.

#include <pthread.h>

int shared_arr[2];

struct S
{
  int x;
  int y;
};
struct S shared_struct;

int cond;

void *writer1(void *_)
{
  shared_arr[0] = 1;       // array-index race target
  shared_struct.x = 1;     // member race target
  if (cond)                // if-guarded race target
    shared_struct.y = 1;
  return 0;
}

void *writer2(void *_)
{
  shared_arr[0] = 2;       // races with writer1's shared_arr[0]
  shared_struct.x = 2;     // races with writer1's shared_struct.x
  if (cond)
    shared_struct.y = 2;   // races with writer1's shared_struct.y
  return 0;
}

int main()
{
  pthread_t t1, t2;
  pthread_create(&t1, 0, writer1, 0);
  pthread_create(&t2, 0, writer2, 0);
  return 0;
}
