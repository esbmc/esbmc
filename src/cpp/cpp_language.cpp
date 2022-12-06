/*******************************************************************\

Module: C++ Language Module

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <ansi-c/c_main.h>
#include <ansi-c/c_preprocess.h>
#include <ansi-c/gcc_builtin_headers.h>
#include <cpp/cpp_final.h>
#include <cpp/cpp_language.h>
#include <cpp/cpp_parser.h>
#include <cpp/cpp_typecheck.h>
#include <util/cpp_expr2string.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include <util/c_link.h>
#include <util/config.h>
#include <util/replace_symbol.h>

bool cpp_languaget::preprocess(const std::string &path, std::ostream &outstream)
{
  if(path == "")
    return c_preprocess("", outstream, true);

  // check extension

  const char *ext = strrchr(path.c_str(), '.');
  if(ext != nullptr && std::string(ext) == ".ipp")
  {
    std::ifstream infile(path.c_str());

    char ch;

    while(infile.read(&ch, 1))
      outstream << ch;

    return false;
  }

  return c_preprocess(path, outstream, true);
}

cpp_languaget::cpp_languaget()
{
}

void cpp_languaget::internal_additions(std::ostream &out)
{
  out << "# 1 \"esbmc_intrinsics.h" << std::endl;

  out << "void *operator new(unsigned int size);" << std::endl;

  out << "extern \"C\" {" << std::endl;

  // assume/assert
  out << "void __ESBMC_assume(bool assumption);" << std::endl;
  out << "void assert(bool assertion);" << std::endl;
  out << "void __ESBMC_assert("
         "bool assertion, const char *description);"
      << std::endl;
  out << "bool __ESBMC_same_object(const void *, const void *);" << std::endl;

  // pointers
  out << "unsigned __ESBMC_POINTER_OBJECT(const void *p);" << std::endl;
  out << "signed __ESBMC_POINTER_OFFSET(const void *p);" << std::endl;

  out << "extern \"C\" int __ESBMC_rounding_mode = 0;" << std::endl;

  // __CPROVER namespace
  out << "namespace __CPROVER { }" << std::endl;

  // for dynamic objects
  out << "unsigned __CPROVER::constant_infinity_uint;" << std::endl;
  out << "bool __ESBMC_alloc[__CPROVER::constant_infinity_uint];" << std::endl;
  out << "unsigned __ESBMC_alloc_size[__CPROVER::constant_infinity_uint];"
      << std::endl;
  out << "bool __ESBMC_is_dynamic[__CPROVER::constant_infinity_uint];"
      << std::endl;

  // float stuff
  out << "bool __ESBMC_isnan(double f);" << std::endl;
  out << "bool __ESBMC_isfinite(double f);" << std::endl;
  out << "bool __ESBMC_isinf(double f);" << std::endl;
  out << "bool __ESBMC_isnormal(double f);" << std::endl;
  out << "int __ESBMC_rounding_mode;" << std::endl;

  // absolute value
  out << "int __ESBMC_abs(int x);" << std::endl;
  out << "long int __ESBMC_labs(long int x);" << std::endl;
  out << "double __ESBMC_fabs(double x);" << std::endl;
  out << "long double __ESBMC_fabsl(long double x);" << std::endl;
  out << "float __ESBMC_fabsf(float x);" << std::endl;

  // Forward decs for pthread main thread begin/end hooks. Because they're
  // pulled in from the C library, they need to be declared prior to pulling
  // them in, for type checking.
  out << "void __ESBMC_pthread_start_main_hook(void);" << std::endl;
  out << "void __ESBMC_pthread_end_main_hook(void);" << std::endl;

  // GCC stuff
  out << GCC_BUILTIN_HEADERS;

  // Forward declarations for nondeterministic types.
  out << "int nondet_int();" << std::endl;
  out << "unsigned int nondet_uint();" << std::endl;
  out << "long nondet_long();" << std::endl;
  out << "unsigned long nondet_ulong();" << std::endl;
  out << "short nondet_short();" << std::endl;
  out << "unsigned short nondet_ushort();" << std::endl;
  out << "short nondet_short();" << std::endl;
  out << "unsigned short nondet_ushort();" << std::endl;
  out << "char nondet_char();" << std::endl;
  out << "unsigned char nondet_uchar();" << std::endl;
  out << "signed char nondet_schar();" << std::endl;
  out << "bool nondet_bool();" << std::endl;
  out << "float nondet_float();" << std::endl;
  out << "double nondet_double();" << std::endl;

  out << "const char *__PRETTY_FUNCTION__;" << std::endl;
  out << "const char *__FILE__ = \"\";" << std::endl;
  out << "unsigned int __LINE__ = 0;" << std::endl;

  out << "}" << std::endl;
}

bool cpp_languaget::parse(const std::string &path)
{
  // store the path

  parse_path = path;

  // preprocessing

  std::ostringstream o_preprocessed;

  internal_additions(o_preprocessed);

  if(preprocess(path, o_preprocessed))
    return true;

  std::istringstream i_preprocessed(o_preprocessed.str());

  // parsing

  cpp_parser.clear();
  cpp_parser.filename = path;
  cpp_parser.in = &i_preprocessed;
  cpp_parser.grammar = cpp_parsert::LANGUAGE;

  if(config.ansi_c.target.is_windows_abi())
    cpp_parser.mode = cpp_parsert::MSC;
  else
    cpp_parser.mode = cpp_parsert::GCC;

  cpp_scanner_init();

  bool result = cpp_parser.parse();

  // save result
  cpp_parse_tree.swap(cpp_parser.parse_tree);

  // save some memory
  cpp_parser.clear();

  return result;
}

bool cpp_languaget::typecheck(contextt &context, const std::string &module)
{
  contextt new_context;

  if(cpp_typecheck(cpp_parse_tree, new_context, module))
    return true;

  return c_link(context, new_context, module);
}

bool cpp_languaget::final(contextt &context)
{
  if(cpp_final(context))
    return true;
  if(c_main(context, "main"))
    return true;

  return false;
}

void cpp_languaget::show_parse(std::ostream &out)
{
  for(cpp_parse_treet::itemst::const_iterator it = cpp_parse_tree.items.begin();
      it != cpp_parse_tree.items.end();
      it++)
    show_parse(out, *it);
}

void cpp_languaget::show_parse(std::ostream &out, const cpp_itemt &item)
{
  if(item.is_linkage_spec())
  {
    const cpp_linkage_spect &linkage_spec = item.get_linkage_spec();

    for(const auto &it : linkage_spec.items())
      show_parse(out, it);

    out << std::endl;
  }
  else if(item.is_namespace_spec())
  {
    const cpp_namespace_spect &namespace_spec = item.get_namespace_spec();

    out << "NAMESPACE " << namespace_spec.get_namespace() << ":" << std::endl;

    for(const auto &it : namespace_spec.items())
      show_parse(out, it);

    out << std::endl;
  }
  else if(item.is_using())
  {
    const cpp_usingt &cpp_using = item.get_using();

    out << "USING ";
    if(cpp_using.get_namespace())
      out << "NAMESPACE ";
    out << cpp_using.name() << std::endl;
    out << std::endl;
  }
  else if(item.is_declaration())
  {
    item.get_declaration().output(out);
  }
  else
    out << "UNKNOWN: " << item << std::endl;
}

languaget *new_cpp_language()
{
  return new cpp_languaget();
}

bool cpp_languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns)
{
  code = cpp_expr2string(expr, ns);
  return false;
}

bool cpp_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns)
{
  code = cpp_type2string(type, ns);
  return false;
}
