#include <jimple-frontend/jimple-language.h>
#include <jimple-frontend/jimple-converter.h>

bool jimple_languaget::typecheck(contextt &context, const std::string &)
{
  log_status("Converting Jimple module {} to GOTO", root.class_name);

  contextt new_context;
  jimple_converter converter(context, root);
  if(converter.convert())
  {
    log_error("Failed to convert module {}", root.class_name);
    return true;
  }

  return false;
}
