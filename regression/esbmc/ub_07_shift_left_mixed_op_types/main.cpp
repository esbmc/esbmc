#include <assert.h>
#include <stdint.h>

int main()
{
  int64_t onel = 1L;
  uint64_t onelu = 1ul;
  int32_t onei = 1;
  uint32_t oneiu = 1u;
  auto result_i_i = 1 << onei;
  assert(result_i_i == 2);
  auto result_i_iu = 1 << oneiu;
  assert(result_i_iu == 2);
  auto result_i_l = 1 << onel;
  assert(result_i_l == 2);
  auto result_i_lu = 1 << onelu;
  assert(result_i_lu == 2);
  auto result_iu_i = 1u << onei;
  assert(result_iu_i == 2u);
  auto result_iu_iu = 1u << oneiu;
  assert(result_iu_iu == 2u);
  auto result_iu_l = 1u << onel;
  assert(result_iu_l == 2u);
  auto result_iu_lu = 1u << onelu;
  assert(result_iu_lu == 2u);
  auto result_l_i = 1l << onei;
  assert(result_l_i == 2l);
  auto result_l_iu = 1l << oneiu;
  assert(result_l_iu == 2l);
  auto result_l_l = 1l << onel;
  assert(result_l_l == 2l);
  auto result_l_lu = 1l << onelu;
  assert(result_l_lu == 2l);
  auto result_lu_i = 1lu << onei;
  assert(result_lu_i == 2lu);
  auto result_lu_iu = 1lu << oneiu;
  assert(result_lu_iu == 2lu);
  auto result_lu_l = 1lu << onel;
  assert(result_lu_l == 2lu);
  auto result_lu_lu = 1lu << onelu;
  assert(result_lu_lu == 2lu);
}
