#include <assert.h>
#include <stdint.h>

int main()
{
  int64_t onel = 64L;
  uint64_t onelu = 64ul;
  int32_t onei = 64;
  uint32_t oneiu = 64u;
  auto result_i_i = 1 << onei;
  auto result_i_iu = 1 << oneiu;
  auto result_i_l = 1 << onel;
  auto result_i_lu = 1 << onelu;
  auto result_iu_i = 1u << onei;
  auto result_iu_iu = 1u << oneiu;
  auto result_iu_l = 1u << onel;
  auto result_iu_lu = 1u << onelu;
  auto result_l_i = 1l << onei;
  auto result_l_iu = 1l << oneiu;
  auto result_l_l = 1l << onel;
  auto result_l_lu = 1l << onelu;
  auto result_lu_i = 1lu << onei;
  auto result_lu_iu = 1lu << oneiu;
  auto result_lu_l = 1lu << onel;
  auto result_lu_lu = 1lu << onelu;
}
