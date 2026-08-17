#include <goto-programs/goto_functions.h>
#include <goto-programs/property_verdict.h>

#include <cstdlib>

property_verdict_tablet goto_functionst::property_verdicts;

property_locationt
property_location(const locationt &location, const std::string &description)
{
  property_locationt loc;
  loc.file = location.get_file().as_string();
  loc.function = location.get_function().as_string();
  loc.description = description;
  loc.line = atoi(location.get_line().c_str());
  loc.column = atoi(location.get_column().c_str());
  return loc;
}

void property_verdict_tablet::record(
  const std::string &property,
  property_verdictt verdict,
  const property_locationt &loc,
  const std::string &note)
{
  if (verdict == property_verdictt::Failed)
    violation = true;

  std::lock_guard lock(mutex);
  auto [it, inserted] =
    results.emplace(property, property_resultt{verdict, note, loc});
  if (!inserted && verdict > it->second.verdict)
    it->second = property_resultt{verdict, note, loc};
}

void property_verdict_tablet::promote_unchecked_to_passed()
{
  std::lock_guard lock(mutex);
  for (auto &[property, result] : results)
    if (result.verdict == property_verdictt::NotChecked)
      result.verdict = property_verdictt::Passed;
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
