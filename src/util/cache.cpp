#include <util/cache.h>
#include <util/message.h>
#include <utility>
#include <util/crypto_hash.h>

namespace
{
std::string hash_irep2(const expr2tc e)
{
  crypto_hash h;
  e->hash(h);
  h.fin();
  return h.to_string();
}
} // namespace
void assertion_cache::run_on_assert(symex_target_equationt::SSA_stept &step)
{
  auto cond_hash = hash_irep2(step.cond);
  auto guard_hash = hash_irep2(step.guard);
  assert_pair pair = std::make_pair(guard_hash, cond_hash);
  auto it = assert_set.find(pair);
  if(it == assert_set.end())
  {
    log_debug("Cache missed");
    assert_set.emplace(pair);
  }
  else
  {
    if((guard_hash == it->first) && (cond_hash == it->second))
    {
      log_debug("Cache hits: {}", ++hits);
      step.cond = constant_bool2tc(trivial_value);
    }
    else
      log_debug("Cache had a hash collision");
  }
}
