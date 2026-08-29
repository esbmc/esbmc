#include <clang-c-frontend/builtin_names.h>
#include <util/config/config.h>

bool compare_float_suffix(const irep_idt &identifier, const std::string &name)
{
  return (identifier == name) || (identifier == (name + "f")) ||
         (identifier == (name + "d")) || (identifier == (name + "l"));
}

bool compare_unscore_builtin(
  const irep_idt &identifier,
  const std::string &name)
{
  const std::string builtin_name = "__builtin_" + name;
  const std::string underscore_name = "__" + name;

  return (identifier == name) ||
         compare_float_suffix(identifier, builtin_name) ||
         (identifier == builtin_name) ||
         compare_float_suffix(identifier, underscore_name) ||
         (identifier == underscore_name);
}

bool is_abs_builtin_name(const irep_idt &identifier)
{
  return identifier == "abs" || identifier == "labs" ||
         identifier == "imaxabs" || identifier == "llabs" ||
         compare_float_suffix(identifier, "fabs") ||
         compare_unscore_builtin(identifier, "fabs");
}

bool is_name_matched_builtin(const irep_idt &identifier)
{
  return is_abs_builtin_name(identifier) ||
         compare_unscore_builtin(identifier, "isnan") ||
         compare_unscore_builtin(identifier, "isinf") ||
         compare_unscore_builtin(identifier, "isnormal") ||
         compare_unscore_builtin(identifier, "signbit") ||
         compare_unscore_builtin(identifier, "isfinite") ||
         compare_float_suffix(identifier, "finite") ||
         compare_unscore_builtin(identifier, "finite") ||
         compare_unscore_builtin(identifier, "inf") ||
         compare_unscore_builtin(identifier, "huge_val");
}

bool builtin_shadows_user_definition(
  const contextt &context,
  const irep_idt &base_name,
  const irep_idt &symbol_id)
{
  if (!is_name_matched_builtin(base_name))
    return false;

  /* c2goto compiles the operational models themselves, where libm/fabs.c and
   * friends do define these names. Those definitions are the models, not a
   * program's, so honouring them here would stop every call inside the models
   * folding to its native node and blow the encoding up (#6904). */
  if (config.options.get_bool_option("building-c-library"))
    return false;

  const symbolt *s = context.find_symbol(symbol_id);
  return s != nullptr && !s->get_value().is_nil();
}
