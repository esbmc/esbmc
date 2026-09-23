#include <goto-programs/goto_functions.h>
#include <goto-programs/property_verdict.h>
#include <util/config/options.h>

#include <cstdint>
#include <cstdlib>

property_verdict_tablet goto_functionst::property_verdicts;

static property_locationt
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

// clear_verified_claims_in_goto turns a checked ASSERT into a SKIP in place,
// keeping its location and number, so a later instance of the claim must key
// the same way off the SKIP.
static bool
is_own_assertion(const goto_programt::instructiont &pc, const std::string &desc)
{
  return (pc.is_assert() || pc.is_skip()) &&
         desc == id2string(pc.location.comment());
}

property_locationt property_location(
  const goto_programt::instructiont &pc,
  const std::string &description)
{
  property_locationt loc = property_location(pc.location, description);
  if (is_own_assertion(pc, description))
  {
    loc.instruction = pc.location_number;
    if (pc.is_assert())
      loc.condition = pc.guard;
  }
  return loc;
}

std::string property_key(
  const goto_programt::instructiont &pc,
  const std::string &description)
{
  const std::string key = description + " at " + pc.location.as_string();
  // The node's address, not its location_number: --bidirectional inserts an
  // ASSERT mid-run and renumbers the program, but no instruction moves.
  return is_own_assertion(pc, description)
           ? key + "\t@" + std::to_string(reinterpret_cast<std::uintptr_t>(&pc))
           : key;
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
  if (inserted)
    return;
  // An instance recorded off a SKIP carries no condition; keep the one taken
  // while the ASSERT was intact.
  expr2tc condition =
    is_nil_expr(loc.condition) ? it->second.loc.condition : loc.condition;
  if (verdict > it->second.verdict)
    it->second = property_resultt{verdict, note, loc};
  it->second.loc.condition = condition;
}

void property_verdict_tablet::promote_unchecked_to_passed()
{
  if (incomplete)
    return;

  std::lock_guard lock(mutex);
  for (auto &[property, result] : results)
    if (result.verdict == property_verdictt::NotChecked)
      result.verdict = property_verdictt::Passed;
}

void property_verdict_tablet::note_incomplete()
{
  incomplete = true;
}

bool withholds_proofs(const optionst &options)
{
  return options.get_bool_option("k-step-property-table") &&
         (options.get_bool_option("base-case") ||
          goto_functionst::property_verdicts.is_incomplete());
}

bool property_verdict_tablet::all_passed() const
{
  std::lock_guard lock(mutex);
  for (const auto &[property, result] : results)
    if (result.verdict != property_verdictt::Passed)
      return false;
  return true;
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
  incomplete = false;
}
