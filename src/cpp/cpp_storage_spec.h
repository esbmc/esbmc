/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_CPP_CPP_STORAGE_SPEC_H
#define CPROVER_CPP_CPP_STORAGE_SPEC_H

#include <location.h>

class cpp_storage_spect:public irept
{
public:
  cpp_storage_spect():irept("cpp-storage-spec")
  {
  }

  locationt &location()
  {
    return static_cast<locationt &>(add("#location"));
  }

  const locationt &location() const
  {
    return static_cast<const locationt &>(find("#location"));
  }
  bool is_static()   const { return get("storage")=="static"; }
  bool is_extern()   const { return get("storage")=="extern"; }
  bool is_auto()     const { return get("storage")=="auto"; }
  bool is_register() const { return get("storage")=="register"; }
  bool is_mutable()  const { return get("storage")=="mutable"; }

  void set_static  () { set("storage", "static"); }
  void set_extern  () { set("storage", "extern"); }
  void set_auto    () { set("storage", "auto"); }
  void set_register() { set("storage", "register"); }
  void set_mutable () { set("storage", "mutable"); }

  bool is_empty() const
  {
    return get("storage")=="";
  }
};

#endif
