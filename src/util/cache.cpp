#include <util/cache.h>
#include <util/message.h>
#include <utility>
void crc_assert_cache::run_on_assert(symex_target_equationt::SSA_stept &step)
{
  auto cond_crc = step.cond->do_crc();
  auto guard_crc = step.guard->do_crc();
  crc_pair pair = std::make_pair(guard_crc, cond_crc);
  auto it = crc_set.find(pair);
  if(it == crc_set.end())
  {
    log_debug("Cache missed");
    crc_set.emplace(pair);
  }
  else
  {
    if((guard_crc == it->first) && (cond_crc == it->second))
    {
      log_debug("Cache hits: {}", ++hits);
      step.cond = constant_bool2tc(trivial_value);
    }
    else
      log_debug("Cache had a hash collision");
  }
}
