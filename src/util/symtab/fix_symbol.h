#include <util/symtab/context.h>
#include <util/symtab/replace_symbol.h>

class fix_symbolt : public replace_symbolt
{
public:
  void fix_symbol(symbolt &symbol);
  void fix_context(contextt &context);
};
