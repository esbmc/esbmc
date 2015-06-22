/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CPP_CPP_MEMBER_SPEC_H
#define CPROVER_CPP_CPP_MEMBER_SPEC_H

#include "location.h"

class cpp_member_spect:public irept
{
public:
  cpp_member_spect():irept("cpp-member-spec")
  {
  }

  bool is_virtual()  const { return get_bool("virtual"); }
  bool is_inline()   const { return get_bool("inline"); }
  bool is_friend()   const { return get_bool("friend"); }
  bool is_explicit() const { return get_bool("explicit"); }

  void set_virtual(bool value)  { set("virtual", value); }
  void set_inline(bool value)   { set("inline", value); }
  void set_friend(bool value)   { set("friend", value); }
  void set_explicit(bool value) { set("explicit", value); }

  bool is_empty() const
  {
    return !is_virtual() &&
           !is_inline() &&
           !is_friend() &&
           !is_explicit();
  }
};

#endif
