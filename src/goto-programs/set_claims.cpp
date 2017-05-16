/*******************************************************************\

Module: Set Claims

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <goto-programs/set_claims.h>
#include <util/i2string.h>

void convert_claims(
  const std::list<std::string> &claims,
  std::set<unsigned> &unsigned_claims)
{
  for(std::list<std::string>::const_iterator
      it=claims.begin();
      it!=claims.end();
      it++)
    unsigned_claims.insert(atoi(it->c_str()));
}

void set_claims(
  goto_programt &goto_program,
  const std::set<unsigned> &claims,
  unsigned &count)
{
  for(goto_programt::instructionst::iterator
      it=goto_program.instructions.begin();
      it!=goto_program.instructions.end();
      it++)
  {
    if(it->is_assert())
    {
      count++;

      if(claims.find(count)==claims.end())
      {
        #if 0
        // assume-guarantee
        if(it->guard.is_true())
          it->type=SKIP;
        else
          it->type=ASSUME;
        #else
        it->type=SKIP;
        #endif
      }
    }
  }
}

void set_claims(
  goto_programt &goto_program,
  const std::list<std::string> &claims)
{
  std::set<unsigned> unsigned_claims;

  convert_claims(claims, unsigned_claims);

  if(unsigned_claims.empty()) return;

  unsigned count=0;

  set_claims(goto_program, unsigned_claims, count);

  unsigned largest=*(--unsigned_claims.end());

  if(count<largest)
    throw "claim "+i2string(largest)+" not found";
}

void set_claims(
  goto_functionst &goto_functions,
  const std::list<std::string> &claims)
{
  std::set<unsigned> unsigned_claims;

  convert_claims(claims, unsigned_claims);

  if(unsigned_claims.empty()) return;

  unsigned count=0;

  for(goto_functionst::function_mapt::iterator
      it=goto_functions.function_map.begin();
      it!=goto_functions.function_map.end();
      it++)
  {
    set_claims(it->second.body, unsigned_claims, count);
  }

  unsigned largest=*(--unsigned_claims.end());

  if(count<largest)
    throw "claim "+i2string(largest)+" not found";
}
