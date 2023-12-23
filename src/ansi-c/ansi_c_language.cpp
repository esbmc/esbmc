/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <ansi-c/ansi_c_convert.h>
#include <ansi-c/ansi_c_language.h>
#include <ansi-c/ansi_c_parser.h>
#include <ansi-c/ansi_c_typecheck.h>
#include <ansi-c/c_final.h>
#include <ansi-c/c_main.h>
#include <ansi-c/c_preprocess.h>
#include <ansi-c/gcc_builtin_headers.h>
#include <ansi-c/trans_unit.h>
#include <util/c_expr2string.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include <util/c_link.h>
#include <util/config.h>
#include <util/expr_util.h>

static void internal_additions(std::string &code)
{
  code +=
    "void __ESBMC_assume(_Bool assumption);\n"
    "void assert(_Bool assertion);\n"
    "void __ESBMC_assert(_Bool assertion, const char *description);\n"
    "_Bool __ESBMC_same_object(const void *, const void *);\n"

    // traces
    "void CBMC_trace(int lvl, const char* event, ...);\n"

    // pointers
    "unsigned __ESBMC_POINTER_OBJECT(const void *p);\n"
    "signed __ESBMC_POINTER_OFFSET(const void *p);\n"

    // malloc / alloca
    "unsigned __ESBMC_constant_infinity_uint;\n"
    "_Bool __ESBMC_alloc[__ESBMC_constant_infinity_uint];\n"
    "_Bool __ESBMC_is_dynamic[__ESBMC_constant_infinity_uint];\n"
    "unsigned long __ESBMC_alloc_size[__ESBMC_constant_infinity_uint];\n"

    // this is ANSI-C
    "extern const char __func__[];\n"

    // float stuff
    "int __ESBMC_rounding_mode = 0;\n"
    "_Bool __ESBMC_floatbv_mode();\n"

    // Forward decs for pthread main thread begin/end hooks. Because they're
    // pulled in from the C library, they need to be declared prior to pulling
    // them in, for type checking.
    "void __ESBMC_pthread_start_main_hook(void);\n"
    "void __ESBMC_pthread_end_main_hook(void);\n"

    // Forward declarations for nondeterministic types.
    "int nondet_int();\n"
    "unsigned int nondet_uint();\n"
    "long nondet_long();\n"
    "unsigned long nondet_ulong();\n"
    "short nondet_short();\n"
    "unsigned short nondet_ushort();\n"
    "char nondet_char();\n"
    "unsigned char nondet_uchar();\n"
    "signed char nondet_schar();\n"
    "_Bool nondet_bool();\n"
    "float nondet_float();\n"
    "double nondet_double();"

    // TACAS definitions,
    "int __VERIFIER_nondet_int();\n"
    "unsigned int __VERIFIER_nondet_uint();\n"
    "long __VERIFIER_nondet_long();\n"
    "unsigned long __VERIFIER_nondet_ulong();\n"
    "short __VERIFIER_nondet_short();\n"
    "unsigned short __VERIFIER_nondet_ushort();\n"
    "char __VERIFIER_nondet_char();\n"
    "unsigned char __VERIFIER_nondet_uchar();\n"
    "signed char __VERIFIER_nondet_schar();\n"
    "_Bool __VERIFIER_nondet_bool();\n"
    "float __VERIFIER_nondet_float();\n"
    "double __VERIFIER_nondet_double();\n"

    "const char *__PRETTY_FUNCTION__;\n"
    "const char *__FILE__ = \"\";\n"
    "unsigned int __LINE__ = 0;\n"

    // GCC junk stuff
    GCC_BUILTIN_HEADERS

    "\n";
}

ansi_c_languaget::ansi_c_languaget()
{
}

bool ansi_c_languaget::preprocess(
  const std::string &path,
  std::ostream &outstream)
{
// check extensions

// TACAS14: preprocess /everything/, including .i files. While the user might
// have preprocessed his file already, we might still want to inject some
// model checker specific stuff into it. A command line option disabling
// preprocessing would be more appropriate.
#if 0
  const char *ext=strrchr(path.c_str(), '.');
  if(ext!=NULL && std::string(ext)==".i")
  {
    std::ifstream infile(path.c_str());

    char ch;

    while(infile.read(&ch, 1))
      outstream << ch;

    return false;
  }
#endif

  return c_preprocess(path, outstream, false);
}

bool ansi_c_languaget::parse(const std::string &path)
{
  // store the path

  parse_path = path;

  // preprocessing

  std::ostringstream o_preprocessed;

  if (preprocess(path, o_preprocessed))
    return true;

  std::istringstream i_preprocessed(o_preprocessed.str());

  // parsing

  std::string code;
  internal_additions(code);
  std::istringstream codestr(code);

  ansi_c_parser.clear();
  ansi_c_parser.filename = "<built-in>";
  ansi_c_parser.in = &codestr;
  //ansi_c_parser.set_message_handler(&message_handler);
  ansi_c_parser.grammar = ansi_c_parsert::LANGUAGE;

  if (config.ansi_c.target.is_windows_abi())
    ansi_c_parser.mode = ansi_c_parsert::MSC;
  else
    ansi_c_parser.mode = ansi_c_parsert::GCC;

  ansi_c_scanner_init();

  bool result = ansi_c_parser.parse();

  if (!result)
  {
    ansi_c_parser.line_no = 0;
    ansi_c_parser.filename = path;
    ansi_c_parser.in = &i_preprocessed;
    ansi_c_scanner_init();
    result = ansi_c_parser.parse();
  }

  // save result
  parse_tree.swap(ansi_c_parser.parse_tree);

  // save some memory
  ansi_c_parser.clear();

  return result;
}

bool ansi_c_languaget::typecheck(contextt &context, const std::string &module)
{
  if (ansi_c_convert(parse_tree, module))
    return true;

  contextt new_context;

  if (ansi_c_typecheck(parse_tree, new_context, module))
    return true;

  if (c_link(context, new_context, module))
    return true;

  return false;
}

bool ansi_c_languaget::final(contextt &context)
{
  if (c_final(context))
    return true;
  if (c_main(context, "main"))
    return true;

  return false;
}

void ansi_c_languaget::show_parse(std::ostream &out)
{
  parse_tree.output(out);
}

languaget *new_ansi_c_language()
{
  return new ansi_c_languaget();
}

bool ansi_c_languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns,
  unsigned flags)
{
  code = c_expr2string(expr, ns, flags);
  return false;
}

bool ansi_c_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns,
  unsigned flags)
{
  code = c_type2string(type, ns, flags);
  return false;
}

bool ansi_c_languaget::merge_context(
  contextt &dest,
  contextt &src,
  const std::string &module) const
{
  return c_link(dest, src, module);
}
