//
// Created by rafaelsamenezes on 23/09/2021.
//

#include <jimple-frontend/jimple-language.h>
#include <jimple-frontend/jimple-converter.h>


bool jimple_languaget::typecheck(
  contextt &context,
  const std::string &,
  const messaget &msg)
{
  msg.status(
    fmt::format("Converting Jimple module {} to GOTO", root.class_name));

  contextt new_context(msg);
  jimple_converter converter(context, root, msg);
  if(converter.convert())
  {
    log_error(fmt::format("Failed to convert module {}", root.class_name));
    return true;
  }

  return false;
}