#include <goto-programs/goto_functions.h>
#include <goto-programs/property_verdict.h>

property_verdict_tablet goto_functionst::property_verdicts;

void property_verdict_tablet::record(
  const std::string &property,
  property_verdictt verdict,
  const std::string &note)
{
  if (verdict == property_verdictt::Failed)
    violation = true;

  std::lock_guard lock(mutex);
  auto [it, inserted] =
    results.emplace(property, property_resultt{verdict, note});
  if (!inserted && verdict > it->second.verdict)
    it->second = property_resultt{verdict, note};
}

std::size_t property_verdict_tablet::size() const
{
  std::lock_guard lock(mutex);
  return results.size();
}

std::map<std::string, property_resultt>
property_verdict_tablet::snapshot() const
{
  std::lock_guard lock(mutex);
  return results;
}

void property_verdict_tablet::clear()
{
  std::lock_guard lock(mutex);
  results.clear();
  violation = false;
}
