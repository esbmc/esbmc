/*******************************************************************\

Module: ANSI-C Linking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <util/context.h>
#include <util/replace_symbol.h>

class fix_symbolt:public replace_symbolt
{
public:
  void fix_symbol(symbolt &symbol);
  void fix_context(contextt &context);
};
