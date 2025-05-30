// "Standards" workaround
#define __STDC_FORMAT_MACROS

#include <smtlib_conv.h>
#include <smtlib.hpp>
#include <smtlib_tok.hpp>

#include <solvers/smt/tuple/smt_tuple_node.h>

#include <cinttypes>
#include <regex>
#include <sstream>

#ifndef _WIN32
#  include <unistd.h>
#  include <signal.h>
#endif

// clang-format off
/** Mapping of SMT function IDs to their names. */
static const std::array smt_func_name_table = {
  // Terminals
  "int_func_id",            /* SMT_FUNC_INT, */
  "bool_func_id",           /* SMT_FUNC_BOOL, */
  "bvint_func_id",          /* SMT_FUNC_BVINT, */
  "real_func_id",           /* SMT_FUNC_REAL, */
  "symbol_func_id",         /* SMT_FUNC_SYMBOL, */

  // Nonterminals
  "+",                      /* SMT_FUNC_ADD, */
  "bvadd",                  /* SMT_FUNC_BVADD, */
  "-",                      /* SMT_FUNC_SUB, */
  "bvsub",                  /* SMT_FUNC_BVSUB, */
  "*",                      /* SMT_FUNC_MUL, */
  "bvmul",                  /* SMT_FUNC_BVMUL, */
  "/",                      /* SMT_FUNC_DIV, */
  "bvudiv",                 /* SMT_FUNC_BVUDIV, */
  "bvsdiv",                 /* SMT_FUNC_BVSDIV, */
  "%",                      /* SMT_FUNC_MOD, */
  "bvsrem",                 /* SMT_FUNC_BVSMOD, */
  "bvurem",                 /* SMT_FUNC_BVUMOD, */
  "shl",                    /* SMT_FUNC_SHL, */
  "bvshl",                  /* SMT_FUNC_BVSHL, */
  "bvashr",                 /* SMT_FUNC_BVASHR, */
  "-",                      /* SMT_FUNC_NEG, */
  "bvneg",                  /* SMT_FUNC_BVNEG, */
  "bvlshr",                 /* SMT_FUNC_BVLSHR, */
  "bvnot",                  /* SMT_FUNC_BVNOT, */
  "bvnxor",                 /* SMT_FUNC_BVNXOR, */
  "bvnor",                  /* SMT_FUNC_BVNOR, */
  "vnand",                  /* SMT_FUNC_BVNAND, */
  "bvxor",                  /* SMT_FUNC_BVXOR, */
  "bvor",                   /* SMT_FUNC_BVOR, */
  "bvand",                  /* SMT_FUNC_BVAND, */

  // Logic
  "=>",                     /* SMT_FUNC_IMPLIES, */
  "xor",                    /* SMT_FUNC_XOR, */
  "or",                     /* SMT_FUNC_OR, */
  "and",                    /* SMT_FUNC_AND, */
  "not",                    /* SMT_FUNC_NOT, */

  // Comparisons
  "<",                      /* SMT_FUNC_LT, */
  "bvslt",                  /* SMT_FUNC_BVSLT, */
  "bvult",                  /* SMT_FUNC_BVULT, */
  ">",                      /* SMT_FUNC_GT, */
  "bvsgt",                  /* SMT_FUNC_BVSGT, */
  "bvugt",                  /* SMT_FUNC_BVUGT, */
  "<=",                     /* SMT_FUNC_LTE, */
  "bvsle",                  /* SMT_FUNC_BVSLTE, */
  "bvule",                  /* SMT_FUNC_BVULTE, */
  ">=",                     /* SMT_FUNC_GTE, */
  "bvsge",                  /* SMT_FUNC_BVSGTE, */
  "bvuge",                  /* SMT_FUNC_BVUGTE, */

  "=",                      /* SMT_FUNC_EQ, */
  "distinct",               /* SMT_FUNC_NOTEQ, */

  "ite",                    /* SMT_FUNC_ITE, */

  "store",                  /* SMT_FUNC_STORE, */
  "select",                 /* SMT_FUNC_SELECT, */

  "concat",                 /* SMT_FUNC_CONCAT, */
  "extract",                /* SMT_FUNC_EXTRACT, */

  "int2real",               /* SMT_FUNC_INT2REAL, */
  "real2int",               /* SMT_FUNC_REAL2INT, */
  "is_int",                 /* SMT_FUNC_IS_INT, */

  // floatbv operations
  "fneg",                   /* SMT_FUNC_FNEG, */
  "fabs",                   /* SMT_FUNC_FABS, */
  "fp.isZero",              /* SMT_FUNC_ISZERO, */
  "fp.isNaN",               /* SMT_FUNC_ISNAN, */
  "fp.isInfinite",          /* SMT_FUNC_ISINF, */
  "fp.isNormal",            /* SMT_FUNC_ISNORMAL, */
  "fp.isNegative",          /* SMT_FUNC_ISNEG, */
  "fp.isPositive",          /* SMT_FUNC_ISPOS, */
  "fp.eq",                  /* SMT_FUNC_IEEE_EQ, */
  "fp.add",                 /* SMT_FUNC_IEEE_ADD, */
  "fp.sub",                 /* SMT_FUNC_IEEE_SUB, */
  "fp.mul",                 /* SMT_FUNC_IEEE_MUL, */
  "fp.div",                 /* SMT_FUNC_IEEE_DIV, */
  "fp.fma",                 /* SMT_FUNC_IEEE_FMA, */
  "fp.sqrt",                /* SMT_FUNC_IEEE_SQRT, */

  "RNE RoundingMode",       /* SMT_FUNC_IEEE_RM_NE, */
  "RTZ RoundingMode",       /* SMT_FUNC_IEEE_RM_ZR, */
  "RTP RoundingMode",       /* SMT_FUNC_IEEE_RM_PI, */
  "RTN RoundingMode",       /* SMT_FUNC_IEEE_RM_MI, */

  "bv2fp_cast",             /* SMT_FUNC_BV2FLOAT, */
  "fp2bv_cast",             /* SMT_FUNC_FLOAT2BV, */
};
// clang-format on

// Dec of external lexer input stream
int smtlibparse(int startval);
extern int smtlib_send_start_code;
extern sexpr *smtlib_output;

#if 0
static std::string unquote(const std::string_view &s)
{
  size_t n = s.size();
  assert(n >= 2);
  assert(s[0] == '"');
  assert(s[n - 1] == '"');
  std::string r;
  r.reserve(n - 2);
  for(size_t i = 1; i < n - 1; i++)
  {
    if(s[i] == '\\')
    {
      assert(i + 1 < n - 1);
      i++;
    }
    r.insert(r.end(), s[i]);
  }
  return r;
}
#endif

smt_convt *create_new_smtlib_solver(
  const optionst &options,
  const namespacet &ns,
  tuple_iface **tuple_api [[maybe_unused]],
  array_iface **array_api,
  fp_convt **fp_api)
{
  if (!options.get_bool_option("smt-formula-only"))
    log_warning(
      "[smtlib] the smtlib interface solving is unstable. Please, "
      "use it with --smt-formula-only for production");
  smtlib_convt *conv = new smtlib_convt(ns, options);
  *array_api = static_cast<array_iface *>(conv);
  *fp_api = static_cast<fp_convt *>(conv);
  return conv;
}

std::string smtlib_convt::dump_smt()
{
  auto path = config.options.get_option("output");
  if (path != "")
  {
    assert(emit_opt_output);
    emit_opt_output.emit("%s\n", "(check-sat)");
    if (path == "-")
      log_status("SMT formula written to standard output");
    else
      log_status("SMT formula written to output file {}", path);
  }
}

smtlib_convt::file_emitter::file_emitter(const std::string &path)
  : out_stream(nullptr)
{
  // We may be being instructed to just output to a file.
  if (path == "")
    return;

  // Open a file, do nothing else.
  out_stream = path == "-" ? stdout : fopen(path.c_str(), "w");
  if (!out_stream)
  {
    log_error("Failed to open \"{}\": {}", path, strerror(errno));
    abort();
  }
}

smtlib_convt::file_emitter::~file_emitter() noexcept
{
  if (out_stream)
    fclose(out_stream);
}

smtlib_convt::process_emitter::process_emitter(const std::string &cmd)
  : out_stream(nullptr), in_stream(nullptr), org_sigpipe_handler(nullptr)
{
  if (cmd == "")
    return;

  // Setup: open a pipe to the smtlib solver. There seems to be no standard C++
  // way of opening a stream from an fd, so use C file streams.

  int inpipe[2], outpipe[2];

#ifdef _WIN32
  // TODO: The current implementation uses UNIX Process
  log_error("smtlib works only in unix systems");
  abort();
#else
  if (pipe(inpipe) != 0)
  {
    log_error("Couldn't open a pipe for smtlib solver");
    abort();
  }

  if (pipe(outpipe) != 0)
  {
    log_error("Couldn't open a pipe for smtlib solver");
    abort();
  }

  pid_t solver_proc_pid = fork();
  if (solver_proc_pid == 0)
  {
    close(outpipe[1]);
    close(inpipe[0]);
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    dup2(outpipe[0], STDIN_FILENO);
    dup2(inpipe[1], STDOUT_FILENO);
    close(outpipe[0]);
    close(inpipe[1]);

    const char *shell = getenv("SHELL");
    if (!shell || !*shell)
      shell = "sh";

    // Voila
    execlp(shell, shell, "-c", cmd.c_str(), NULL);
    log_error(
      "Exec of smtlib solver failed: {} -c {}: {}",
      shell,
      cmd,
      strerror(errno));
    abort();
  }
  else
  {
    close(outpipe[0]);
    close(inpipe[1]);
    out_stream = fdopen(outpipe[1], "w");
    in_stream = fdopen(inpipe[0], "r");

    org_sigpipe_handler = reinterpret_cast<void *>(signal(SIGPIPE, SIG_IGN));
    if (org_sigpipe_handler == SIG_ERR)
    {
      log_error("registering SIGPIPE handler: {}", strerror(errno));
      abort();
    }
  }
  // Execution continues as the parent ESBMC process. Child dying will
  // trigger SIGPIPE or an EOF eventually, which we'll be able to detect
  // and crash upon.

  // Point lexer input at output stream
  smtlib_tokin = in_stream;

  // Fetch solver name and version.
#  if 0
  emit("%s", "(get-info :name)\n");
  flush();
  smtlib_send_start_code = 1;
  smtlibparse(TOK_START_INFO);

  // As a result we should have a single entry in a list of sexprs.
  class sexpr *sexpr = smtlib_output;
  assert(
    sexpr->sexpr_list.size() == 1 &&
    "More than one sexpr response to get-info name");
  class sexpr &s = sexpr->sexpr_list.front();

  // Should have a keyword followed by a string?
  assert(s.token == 0 && s.sexpr_list.size() == 2 && "Bad solver name format");
  class sexpr &keyword = s.sexpr_list.front();
  class sexpr &value = s.sexpr_list.back();
  if(!(keyword.token == TOK_KEYWORD && keyword.data == ":name"))
  {
    log_error("Bad get-info :name response from solver");
    abort();
  }

  assert(value.token == TOK_STRINGLIT && "Non-string solver name response");
  solver_name = unquote(value.data);
  delete smtlib_output;

  // Duplicate / boilerplate;
  emit("%s", "(get-info :version)\n");
  flush();
  smtlib_send_start_code = 1;
  smtlibparse(TOK_START_INFO);

  sexpr = smtlib_output;
  assert(
    sexpr->sexpr_list.size() == 1 &&
    "More than one sexpr response to get-info version");
  class sexpr &v = sexpr->sexpr_list.front();

  if(v.token == 0 && v.sexpr_list.size() != 2)
  {
    log_error("Bad solver version fmt");
    abort();
  }
  class sexpr &kw = v.sexpr_list.front();
  class sexpr &val = v.sexpr_list.back();
  if(!(kw.token == TOK_KEYWORD && kw.data == ":version"))
    std::runtime_error("Bad get-info :version response from solver");

  assert(val.token == TOK_STRINGLIT && "Non-string solver version response");
  solver_version = unquote(val.data);
  delete smtlib_output;

  log_status(
    "Using external solver '{}' version '{}' with PID {}",
    solver_name,
    solver_version,
    solver_proc_pid);
#  else
  log_status(
    "Using external solver cmd '{}' with PID {}", cmd, solver_proc_pid);
#  endif
#endif
}

smtlib_convt::process_emitter::~process_emitter() noexcept
{
  if (out_stream)
    fclose(out_stream);
  if (in_stream)
    fclose(in_stream);
#ifndef _WIN32
  if (org_sigpipe_handler)
    signal(SIGPIPE, reinterpret_cast<void (*)(int)>(org_sigpipe_handler));
#endif
}

smtlib_convt::smtlib_convt(const namespacet &_ns, const optionst &_options)
  : smt_convt(_ns, _options),
    array_iface(true, false),
    fp_convt(this),
    emit_proc(_options.get_option("smtlib-solver-prog")),
    emit_opt_output(_options.get_option("output"))
{
  std::string logic =
    options.get_bool_option("int-encoding") ? "QF_AUFLIRA" : "QF_AUFBV";

  emit("%s", "(set-option :produce-models true)\n");
  emit("(set-logic %s)\n", logic.c_str());
  emit("%s", "(set-info :status unknown)\n");
}

smtlib_convt::~smtlib_convt()
{
  delete_all_asts();
}

std::string smtlib_convt::sort_to_string(const smt_sort *s) const
{
  const smtlib_smt_sort *sort = static_cast<const smtlib_smt_sort *>(s);
  std::stringstream ss;

  switch (sort->id)
  {
  case SMT_SORT_INT:
    return "Int";
  case SMT_SORT_REAL:
    return "Real";
  case SMT_SORT_FIXEDBV:
  case SMT_SORT_BV:
  case SMT_SORT_BVFP:
    ss << "(_ BitVec " << sort->get_data_width() << ")";
    return ss.str();
  case SMT_SORT_ARRAY:
    ss << "(Array " << sort_to_string(sort->domain) << " "
       << sort_to_string(sort->range) << ")";
    return ss.str();
  case SMT_SORT_BOOL:
    return "Bool";
  default:
    log_error("Unexpected sort in smtlib_convt");
    abort();
  }
}

/* TODO: misnomer, it does not emit anything */
unsigned int smtlib_convt::emit_terminal_ast(
  const smtlib_smt_ast *ast,
  std::string &output) const
{
  std::stringstream ss;
  const smtlib_smt_sort *sort = static_cast<const smtlib_smt_sort *>(ast->sort);

  switch (ast->kind)
  {
  case SMT_FUNC_INT:
    if (ast->intval.is_negative())
    {
      // Negative integers need to be constructed from unary minus and a literal
      ss << "(- " << integer2string(-ast->intval) << ")";
      output = ss.str();
    }
    else
    {
      // Just the literal number itself.
      output = integer2string(ast->intval);
    }
    return 0;
  case SMT_FUNC_BOOL:
    if (ast->boolval)
      output = "true";
    else
      output = "false";
    return 0;
  case SMT_FUNC_BVINT:
    // Construct a bitvector
    {
      size_t n = sort->get_data_width();
      assert(n > 0);
      /* Two's complement, n bits wide */
      ss << "#b" << integer2binary(ast->intval, n);
      output = ss.str();
      return 0;
    }
  case SMT_FUNC_REAL:
    // Give up
    ss << ast->realval;
    output = ss.str();
    return 0;
  case SMT_FUNC_SYMBOL:
  {
    /* from smt-lib 2.6:
     * A quoted symbol is any sequence of whitespace characters and printable
     * characters that starts and ends with | and does not contain | or \ */

    /* All symbols to be emitted as quoted symbols (braced within |'s),
     * therefore replace (in order):
     *   / -> //
     *   \ -> /b
     *   | -> /p
     */
    std::string replaced = ast->symname;
    replaced = std::regex_replace(replaced, std::regex("/"), "//");
    replaced = std::regex_replace(replaced, std::regex("\\\\"), "/b");
    replaced = std::regex_replace(replaced, std::regex("\\|"), "/p");

    ss << "|" << replaced << "|";
    output = ss.str();
    return 0;
  }
  default:
    log_error("Invalid terminal AST kind");
    abort();
  }
}

unsigned int smtlib_convt::emit_ast(
  const smtlib_smt_ast *ast,
  std::string &output,
  std::unordered_map<const smtlib_smt_ast *, std::string> &temp_symbols) const
{
  unsigned int brace_level = 0;
  assert(ast->args.size() <= 4);
  std::string args[4];

  switch (ast->kind)
  {
  case SMT_FUNC_INT:
  case SMT_FUNC_BOOL:
  case SMT_FUNC_BVINT:
  case SMT_FUNC_REAL:
  case SMT_FUNC_SYMBOL:
    return emit_terminal_ast(ast, output);
  default:
    break;
    // Continue.
  }

  if (auto it = temp_symbols.find(ast); it != temp_symbols.end())
  {
    output = it->second;
    return 0;
  }

  // Get a temporary sym name
  size_t tempnum = temp_symbols.size();
  std::stringstream ss;
  ss << "?x" << tempnum;
  std::string tempname = ss.str();

  temp_symbols.emplace(ast, tempname);

  for (unsigned long int i = 0; i < ast->args.size(); i++)
    brace_level += emit_ast(
      static_cast<const smtlib_smt_ast *>(ast->args[i]), args[i], temp_symbols);

  // Emit a let, assigning the result of this AST func to the sym.
  // For some reason let requires a double-braced operand.
  emit("(let ((%s (", tempname.c_str());

  // This asts function
  assert(static_cast<size_t>(ast->kind) < smt_func_name_table.size());
  if (ast->kind == SMT_FUNC_EXTRACT)
  {
    // Extract is an indexed function
    emit("(_ extract %d %d)", ast->extract_high, ast->extract_low);
  }
  else
  {
    emit("%s", smt_func_name_table[ast->kind]);
  }

  // Its operands
  for (unsigned long int i = 0; i < ast->args.size(); i++)
    emit(" %s", args[i].c_str());

  // End func enclosing brace, then operand to let (two braces).
  emit("%s", ")))\n");

  // We end with one additional brace level.
  output = tempname;
  return brace_level + 1;
}

void smtlib_convt::emit_ast(const smtlib_smt_ast *ast) const
{
  // The algorithm: descend through the AST operands, binding values to
  // temporary symbols, then emit functions on those temporary symbols.
  // All recursively. The non-trivial bit is tracking how many ending
  // braces are required.
  // This is inspired by the output from Z3 that I've seen.

  std::string output;
  std::unordered_map<const smtlib_smt_ast *, std::string> temp_symbols;
  unsigned int brace_level = emit_ast(ast, output, temp_symbols);

  /* Emit the final representation of the root, either a (possibly temporary)
   * symbol, or that of a terminal. */
  emit("%s", output.c_str());

  // Emit a ton of end braces.
  for (unsigned int i = 0; i < brace_level; i++)
    emit("%c", ')');
}

void smtlib_smt_ast::dump() const
{
  const smtlib_convt *ctx = static_cast<const smtlib_convt *>(context);

  /* XXX fbrausse: Hack. No worries though, the context is dynamically allocated
   * and we're restoring its state at the end of this function. */
  smtlib_convt *ctx_m = const_cast<smtlib_convt *>(ctx);
  FILE *tmp_file = std::exchange(ctx_m->emit_opt_output.out_stream, stderr);
  FILE *tmp_proc = std::exchange(ctx_m->emit_proc.out_stream, nullptr);

  ctx->emit_ast(this);
  ctx->emit("%s", "\n");
  std::string sort_str = ctx->sort_to_string(sort);
  ctx->emit("sort: %s\n", sort_str.c_str());
  ctx->flush();

  ctx_m->emit_opt_output.out_stream = tmp_file;
  ctx_m->emit_proc.out_stream = tmp_proc;
}

smt_convt::resultt smtlib_convt::dec_solve()
{
  pre_solve();

  // Set some preliminaries, logic and so forth.
  // Declare all the symbols + sorts
  // Emit constraints
  // check-sat

  emit("%s", "(check-sat)\n");

  // Flush out command, starting model check
  flush();

  // If we're just outputing to a file, this is where we terminate.
  if (!emit_proc)
    return smt_convt::P_SMTLIB;

  // And read in the output
  smtlib_send_start_code = 1;
  smtlibparse(TOK_START_SAT);

  // This should generate on sexpr. See what it is.
  if (smtlib_output->token == TOK_KW_SAT)
  {
    return smt_convt::P_SATISFIABLE;
  }
  if (smtlib_output->token == TOK_KW_UNSAT)
  {
    return smt_convt::P_UNSATISFIABLE;
  }
  else if (smtlib_output->token == TOK_KW_ERROR)
  {
    log_error("SMTLIB solver returned: \"{}\"", smtlib_output->data);
    return smt_convt::P_ERROR;
  }
  else
  {
    log_error("Unrecognized check-sat output from smtlib solver");
    abort();
  }
}

sexpr smtlib_convt::get_value(smt_astt a) const
{
  assert(emit_proc);

  emit("%s", "(get-value (");
  emit_ast(to_solver_smt_ast<smtlib_smt_ast>(a));
  emit("%s\n", "))");
  flush();
  smtlib_send_start_code = 1;
  smtlibparse(TOK_START_VALUE);

  if (smtlib_output->token == TOK_KW_ERROR)
  {
    log_error(
      "Error from smtlib solver when fetching literal value: \"{}\"",
      smtlib_output->data);
    abort();
  }
  else if (smtlib_output->token != 0)
  {
    log_error("Unrecognized response to get-value from smtlib solver");
    abort();
  }
  // Unpack our value from response list.
  assert(
    smtlib_output->sexpr_list.size() == 1 &&
    "More than one response to "
    "get-value from smtlib solver");
  sexpr &response = *smtlib_output->sexpr_list.begin();
  // Now we have a valuation pair. First is the symbol
  assert(
    response.sexpr_list.size() == 2 &&
    "Expected 2 operands in "
    "valuation_pair_list from smtlib solver");
  std::list<sexpr>::iterator it = response.sexpr_list.begin();
  /* sexpr &symname = *it; */
  sexpr respval = std::move(*++it);

  delete smtlib_output;
  return respval;
}

static BigInt interp_numeric(const sexpr &respval, bool is_signed)
{
  yytokentype tok = static_cast<yytokentype>(respval.token);
  switch (tok)
  {
  case TOK_DECIMAL:
    return string2integer(respval.data);
  case TOK_HEXNUM:
    return string2integer(respval.data.substr(2), 16);
  case TOK_BINNUM:
    return binary2integer(respval.data.substr(2), is_signed);
  default:
    log_error(
      "interpreting S-expr of token type {} as an integer",
      fmt::underlying(tok));
    abort();
  }
}

BigInt smtlib_convt::get_bv(smt_astt a, bool is_signed)
{
  sexpr respval = get_value(a);

  // Attempt to read an integer.
  BigInt m = interp_numeric(respval, is_signed);

  return m;
}

expr2tc
smtlib_convt::get_array_elem(smt_astt array, uint64_t index, const type2tc &t)
{
  assert(emit_proc);

  uint64_t domain_width = array->sort->get_domain_width();
  smt_astt sel =
    array->select(this, constant_int2tc(get_uint_type(domain_width), index));

  return get_by_ast(t, sel);
}

static std::string read_all(FILE *in)
{
  std::string r;
  char buf[4096];
  for (size_t rd; (rd = fread(buf, 1, sizeof(buf), in));)
    r.insert(r.size(), buf, rd);
  return r;
}

template <typename... Ts>
void smtlib_convt::emit(const Ts &...ts) const
{
  if (emit_proc)
    emit_proc.emit(ts...);
  if (emit_opt_output)
    emit_opt_output.emit(ts...);
}

void smtlib_convt::flush() const
{
  if (emit_proc)
    emit_proc.flush();
  if (emit_opt_output)
    emit_opt_output.flush();
}

smtlib_convt::process_emitter::operator bool() const noexcept
{
  return out_stream != nullptr;
}

template <typename... Ts>
void smtlib_convt::process_emitter::emit(const char *fmt, Ts &&...ts) const
{
  /* TODO: other error handling */
  errno = 0;
  if (fprintf(out_stream, fmt, ts...) < 0 && errno == EPIPE)
    throw external_process_died(read_all(in_stream));
}

void smtlib_convt::process_emitter::flush() const
{
  /* TODO: other error handling */
  errno = 0;
  if (fflush(out_stream) == EOF && errno == EPIPE)
    throw external_process_died(read_all(in_stream));
}

smtlib_convt::file_emitter::operator bool() const noexcept
{
  return out_stream != nullptr;
}

template <typename... Ts>
void smtlib_convt::file_emitter::emit(const char *fmt, Ts &&...ts) const
{
  /* TODO: error handling */
  fprintf(out_stream, fmt, ts...);
}

void smtlib_convt::file_emitter::flush() const
{
  /* TODO: error handling */
  fflush(out_stream);
}

tvt smtlib_convt::l_get(smt_astt a)
{
  sexpr second = get_value(a);

  // And finally we have our value. It should be true or false.
  if (second.token == TOK_KW_TRUE)
    return tvt(true);
  if (second.token == TOK_KW_FALSE)
    return tvt(false);

  if (second.token == TOK_SIMPLESYM && second.data == "???")
  {
    /* Yices sometimes returns '???', e.g. when using get-value of stores */
    return tvt(tvt::TV_UNKNOWN);
  }

  /* Boolector sometimes returns #b0 or #b1 for Bool-sorted constants */
  BigInt m = interp_numeric(second, false);
  if (m == 0)
    return tvt(false);
  if (m == 1)
    return tvt(true);

  abort();
}

bool smtlib_convt::get_bool(smt_astt a)
{
  tvt tv = l_get(a);

  if (tv.is_true())
    return true;
  if (tv.is_false())
    return false;

  abort();
}

const std::string smtlib_convt::solver_text()
{
  if (emit_proc)
    return "'" + options.get_option("smtlib-solver-prog") + "'";

  if (emit_opt_output)
    return "Text output";

  return "<smtlib:none>";
}

void smtlib_convt::assert_ast(smt_astt a)
{
  const smtlib_smt_ast *sa = static_cast<const smtlib_smt_ast *>(a);

  // Encode an assertion
  emit("%s", "(assert\n");

  emit_ast(sa);

  // Final brace for closing the 'assert'.
  emit("%s", ")\n");
}

smt_astt smtlib_convt::mk_smt_int(const BigInt &theint)
{
  smt_sortt s = mk_int_sort();
  smtlib_smt_ast *a = new smtlib_smt_ast(this, s, SMT_FUNC_INT);
  a->intval = theint;
  return a;
}

smt_astt smtlib_convt::mk_smt_real(const std::string &str)
{
  smt_sortt s = mk_real_sort();
  smtlib_smt_ast *a = new smtlib_smt_ast(this, s, SMT_FUNC_REAL);
  a->realval = str;
  return a;
}

smt_astt smtlib_convt::mk_smt_bv(const BigInt &theint, smt_sortt s)
{
  smtlib_smt_ast *a = new smtlib_smt_ast(this, s, SMT_FUNC_BVINT);
  a->intval = theint;
  return a;
}

smt_astt smtlib_convt::mk_smt_bool(bool val)
{
  smtlib_smt_ast *a = new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_BOOL);
  a->boolval = val;
  return a;
}

smt_astt smtlib_convt::mk_array_symbol(
  const std::string &name,
  const smt_sort *s,
  smt_sortt array_subtype [[maybe_unused]])
{
  return mk_smt_symbol(name, s);
}

smt_astt smtlib_convt::mk_smt_symbol(const std::string &name, const smt_sort *s)
{
  smtlib_smt_ast *a = new smtlib_smt_ast(this, s, SMT_FUNC_SYMBOL);
  a->symname = name;

  symbol_tablet::iterator it = symbol_table.find(name);

  if (it != symbol_table.end())
    return a;

  // Record the type of this symbol
  struct symbol_table_rec record = {name, ctx_level, s};
  symbol_table.insert(record);

  if (s->id == SMT_SORT_STRUCT)
    return a;

  // As this is the first time, declare that symbol to the solver.
  std::string output;
  emit_terminal_ast(a, output);
  emit("(declare-fun %s () %s)\n", output.c_str(), sort_to_string(s).c_str());

  return a;
}

smt_sort *smtlib_convt::mk_struct_sort(const type2tc &type [[maybe_unused]])
{
  log_error("Attempted to make struct type in smtlib conversion");
  abort();
}

smt_astt
smtlib_convt::mk_extract(smt_astt a, unsigned int high, unsigned int low)
{
  smt_sortt s = mk_bv_sort(high - low + 1);
  smtlib_smt_ast *n = new smtlib_smt_ast(this, s, SMT_FUNC_EXTRACT);
  n->extract_high = high;
  n->extract_low = low;
  n->args.push_back(a);
  return n;
}

smt_astt smtlib_convt::mk_sign_ext(smt_astt a, unsigned int topwidth)
{
  std::size_t topbit = a->sort->get_data_width();
  smt_astt the_top_bit = mk_extract(a, topbit - 1, topbit - 1);
  smt_astt zero_bit = mk_smt_bv(0, mk_bv_sort(1));
  smt_astt t = mk_eq(the_top_bit, zero_bit);

  smt_astt z = mk_smt_bv(0, mk_bv_sort(topwidth));

  // Calculate the exact value; SMTLIB text parsers don't like taking an
  // over-full integer literal.
  uint64_t big = 0xFFFFFFFFFFFFFFFFULL;
  unsigned int num_topbits = 64 - topwidth;
  big >>= num_topbits;
  smt_astt f = mk_smt_bv(big, mk_bv_sort(topwidth));

  smt_astt topbits = mk_ite(t, z, f);

  return mk_concat(topbits, a);
}

smt_astt smtlib_convt::mk_zero_ext(smt_astt a, unsigned int topwidth)
{
  log_debug("smtlib", "[smt_ast] mk_zero_ext with {} width", topwidth);
  smt_astt z = mk_smt_bv(0, mk_bv_sort(topwidth));
  return mk_concat(z, a);
}

smt_astt smtlib_convt::mk_concat(smt_astt a, smt_astt b)
{
  /**
   * (concat (_ BitVec i) (_ BitVec j) (_ BitVec m))
      - concatenation of bitvectors of size i and j to get a new bitvector of
        size m, where m = i + j
  */
  smtlib_smt_ast *ast = new smtlib_smt_ast(
    this,
    mk_bv_sort(a->sort->get_data_width() + b->sort->get_data_width()),
    SMT_FUNC_CONCAT);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_ite(smt_astt cond, smt_astt t, smt_astt f)
{
  assert(cond->sort->id == SMT_SORT_BOOL);
  assert(t->sort->get_data_width() == f->sort->get_data_width());

  smtlib_smt_ast *ast = new smtlib_smt_ast(this, t->sort, SMT_FUNC_ITE);
  ast->args.push_back(cond);
  ast->args.push_back(t);
  ast->args.push_back(f);
  return ast;
}

int smtliberror(int startsym [[maybe_unused]], const std::string &error)
{
  log_error("SMTLIB response parsing: \"{}\"", error);
  abort();
}

void smtlib_convt::push_ctx()
{
  smt_convt::push_ctx();

  emit("%s", "(push 1)\n");
}

smt_astt smtlib_convt::mk_add(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_ADD);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvadd(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_BVADD);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_sub(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_SUB);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvsub(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_BVSUB);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_mul(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_MUL);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvmul(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_BVMUL);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_mod(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_MOD);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvsmod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_BVSMOD);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvumod(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_BVUMOD);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_div(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_DIV);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvsdiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_BVSDIV);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvudiv(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_BVUDIV);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_shl(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  assert(a->sort->id == b->sort->id);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_SHL);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvshl(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_BVSHL);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvashr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_BVASHR);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvlshr(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_BVLSHR);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_neg(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_NEG);
  ast->args.push_back(a);
  return ast;
}

smt_astt smtlib_convt::mk_bvneg(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_BVNEG);
  ast->args.push_back(a);
  return ast;
}

smt_astt smtlib_convt::mk_bvnot(smt_astt a)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_BVNOT);
  ast->args.push_back(a);
  return ast;
}

smt_astt smtlib_convt::mk_bvnxor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_BVNXOR);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvnor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_BVNOR);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvnand(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_BVNAND);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvxor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_BVXOR);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvor(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_BVOR);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvand(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_BVAND);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_implies(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  smtlib_smt_ast *ast =
    new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_IMPLIES);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_xor(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_XOR);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_or(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_OR);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_and(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_BOOL && b->sort->id == SMT_SORT_BOOL);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_AND);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_not(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_BOOL);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_NOT);
  ast->args.push_back(a);
  return ast;
}

smt_astt smtlib_convt::mk_lt(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_LT);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvult(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_BVULT);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvslt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_BVSLT);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_gt(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_GT);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvugt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_BVUGT);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvsgt(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_BVSGT);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_le(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_LTE);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvule(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_BVULTE);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvsle(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_BVSLTE);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_ge(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  assert(b->sort->id == SMT_SORT_INT || b->sort->id == SMT_SORT_REAL);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_GTE);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvuge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_BVUGTE);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_bvsge(smt_astt a, smt_astt b)
{
  assert(a->sort->id != SMT_SORT_INT && a->sort->id != SMT_SORT_REAL);
  assert(b->sort->id != SMT_SORT_INT && b->sort->id != SMT_SORT_REAL);
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_BVSGTE);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_eq(smt_astt a, smt_astt b)
{
  assert(a->sort->get_data_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_EQ);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_store(smt_astt a, smt_astt b, smt_astt c)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  assert(
    a->sort->get_range_sort()->get_data_width() == c->sort->get_data_width());
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_STORE);
  ast->args.push_back(a);
  ast->args.push_back(b);
  ast->args.push_back(c);
  return ast;
}

smt_astt smtlib_convt::mk_select(smt_astt a, smt_astt b)
{
  assert(a->sort->id == SMT_SORT_ARRAY);
  assert(a->sort->get_domain_width() == b->sort->get_data_width());
  smtlib_smt_ast *ast =
    new smtlib_smt_ast(this, a->sort->get_range_sort(), SMT_FUNC_SELECT);
  ast->args.push_back(a);
  ast->args.push_back(b);
  return ast;
}

smt_astt smtlib_convt::mk_real2int(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_REAL);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_REAL2INT);
  ast->args.push_back(a);
  return ast;
}

smt_astt smtlib_convt::mk_int2real(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_INT);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, a->sort, SMT_FUNC_INT2REAL);
  ast->args.push_back(a);
  return ast;
}

smt_astt smtlib_convt::mk_isint(smt_astt a)
{
  assert(a->sort->id == SMT_SORT_INT || a->sort->id == SMT_SORT_REAL);
  smtlib_smt_ast *ast = new smtlib_smt_ast(this, boolean_sort, SMT_FUNC_IS_INT);
  ast->args.push_back(a);
  return ast;
}

void smtlib_convt::pop_ctx()
{
  emit("%s", "(pop 1)\n");

  // Wipe this level of symbol table.
  symbol_tablet::nth_index<1>::type &syms_numindex = symbol_table.get<1>();
  syms_numindex.erase(ctx_level);

  smt_convt::pop_ctx();
}

smt_astt
smtlib_convt::convert_array_of(smt_astt init_val, unsigned long domain_width)
{
  return default_convert_array_of(init_val, domain_width, this);
}

smt_sortt smtlib_convt::mk_bool_sort()
{
  return new smtlib_smt_sort(SMT_SORT_BOOL);
}

smt_sortt smtlib_convt::mk_real_sort()
{
  return new smtlib_smt_sort(SMT_SORT_REAL);
}

smt_sortt smtlib_convt::mk_int_sort()
{
  return new smtlib_smt_sort(SMT_SORT_INT);
}

smt_sortt smtlib_convt::mk_bv_sort(std::size_t width)
{
  return new smtlib_smt_sort(SMT_SORT_BV, width);
}

smt_sortt smtlib_convt::mk_fbv_sort(std::size_t width)
{
  return new smtlib_smt_sort(SMT_SORT_FIXEDBV, width);
}

smt_sortt smtlib_convt::mk_array_sort(smt_sortt domain, smt_sortt range)
{
  const smtlib_smt_sort *d = static_cast<const smtlib_smt_sort *>(domain);
  const smtlib_smt_sort *r = static_cast<const smtlib_smt_sort *>(range);
  return new smtlib_smt_sort(SMT_SORT_ARRAY, d, r);
}

smt_sortt smtlib_convt::mk_bvfp_sort(std::size_t ew, std::size_t sw)
{
  return new smtlib_smt_sort(SMT_SORT_BVFP, ew + sw + 1, sw + 1);
}

smt_sortt smtlib_convt::mk_bvfp_rm_sort()
{
  return new smtlib_smt_sort(SMT_SORT_BVFP_RM, 3);
}
