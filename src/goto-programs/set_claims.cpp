#include <goto-programs/set_claims.h>
#include <util/i2string.h>

void convert_claims(
  const std::list<std::string> &claims,
  std::set<unsigned> &unsigned_claims)
{
  for(const auto &claim : claims)
    unsigned_claims.insert(atoi(claim.c_str()));
}

void set_claims(
  goto_programt &goto_program,
  const std::set<unsigned> &claims,
  unsigned &count)
{
  for(auto &instruction : goto_program.instructions)
  {
    if(instruction.is_assert())
    {
      count++;

      if(claims.find(count) == claims.end())
        instruction.make_skip();
    }
  }
}

void set_claims(
  goto_programt &goto_program,
  const std::list<std::string> &claims)
{
  std::set<unsigned> unsigned_claims;

  convert_claims(claims, unsigned_claims);

  if(unsigned_claims.empty())
    return;

  unsigned count = 0;

  set_claims(goto_program, unsigned_claims, count);

  unsigned largest = *(--unsigned_claims.end());

  if(count < largest)
    throw "claim " + i2string(largest) + " not found";
}

void set_claims(
  goto_functionst &goto_functions,
  const std::list<std::string> &claims)
{
  std::set<unsigned> unsigned_claims;
  convert_claims(claims, unsigned_claims);

  if(unsigned_claims.empty())
    return;

  unsigned count = 0;
  for(auto &it : goto_functions.function_map)
  {
    if(!it.second.body.empty())
      set_claims(it.second.body, unsigned_claims, count);
  }

  unsigned largest = *(--unsigned_claims.end());

  if(count < largest)
    throw "claim " + i2string(largest) + " not found";
}
