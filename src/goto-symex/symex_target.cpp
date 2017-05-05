/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <goto-symex/symex_target.h>

bool operator < (const symex_targett::sourcet &a, const symex_targett::sourcet &b)
{
  if(a.thread_nr < b.thread_nr) return true;
  if(a.thread_nr > b.thread_nr) return false;
  return a.pc < b.pc;
}
