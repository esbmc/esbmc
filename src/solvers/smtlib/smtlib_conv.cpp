// "Standards" workaround
#define __STDC_FORMAT_MACROS

#include <cinttypes>
#include <smtlib_conv.h>
#include <smtlib.hpp>
#include <smtlib_tok.hpp>
#include <sstream>
#include <unistd.h>

// Dec of external lexer input stream
int smtlibparse(int startval);
extern int smtlib_send_start_code;
extern sexpr *smtlib_output;

smt_convt *
create_new_smtlib_solver(bool int_encoding, const namespacet &ns,
                          const optionst &opts __attribute__((unused)),
                          tuple_iface **tuple_api __attribute__((unused)),
                          array_iface **array_api)
{
  smtlib_convt *conv = new smtlib_convt(int_encoding, ns, opts);
  *array_api = static_cast<array_iface*>(conv);
  return conv;
}

smtlib_convt::smtlib_convt(bool int_encoding, const namespacet &_ns,
                           const optionst &_opts)
  : smt_convt(int_encoding, _ns), array_iface(false, false),
    options(_opts)
{

  temp_sym_count.push_back(1);
  std::string cmd;

  std::string logic = (int_encoding) ? "QF_AUFLIRA" : "QF_AUFBV";

  // We may be being instructed to just output to a file.
  cmd = options.get_option("output");
  if (cmd != "") {
    if (options.get_option("smtlib-solver-prog") != "") {
      std::cerr << "Can't solve SMTLIB output and write to a file, sorry"
                << std::endl;
      abort();
    }

    // Open a file, do nothing else.
    out_stream = fopen(cmd.c_str(), "w");
    if (!out_stream) {
      std::cerr << "Failed to open \"" << cmd << "\"" << std::endl;
      abort();
    }

    in_stream = NULL;
    solver_name = "Text output";
    solver_version = "";
    solver_proc_pid = 0;

    fprintf(out_stream, "(set-logic %s)\n", logic.c_str());
    fprintf(out_stream, "(set-info :status unknown)\n");
    fprintf(out_stream, "(set-option :produce-models true)\n");

    return;
  }

  // Setup: open a pipe to the smtlib solver. There seems to be no standard C++
  // way of opening a stream from an fd, so use C file streams.

  int inpipe[2], outpipe[2];

  cmd = options.get_option("smtlib-solver-prog");
  if (cmd == "") {
    std::cerr << "Must specify an smtlib solver program in smtlib mode"
              << std::endl;
    abort();
  }

  if (pipe(inpipe) != 0) {
    std::cerr << "Couldn't open a pipe for smtlib solver" << std::endl;
    abort();
  }

  if (pipe(outpipe) != 0) {
    std::cerr << "Couldn't open a pipe for smtlib solver" << std::endl;
    abort();
  }

  solver_proc_pid = fork();
  if (solver_proc_pid == 0) {
    close(outpipe[1]);
    close(inpipe[0]);
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    dup2(outpipe[0], STDIN_FILENO);
    dup2(inpipe[1], STDOUT_FILENO);
    close(outpipe[0]);
    close(inpipe[1]);

    // Voila
    execlp(cmd.c_str(), cmd.c_str(), NULL);
    std::cerr << "Exec of smtlib solver failed" << std::endl;
    abort();
  } else {
    close(outpipe[0]);
    close(inpipe[1]);
    out_stream = fdopen(outpipe[1], "w");
    in_stream = fdopen(inpipe[0], "r");
  }

  // Execution continues as the parent ESBMC process. Child dying will
  // trigger SIGPIPE or an EOF eventually, which we'll be able to detect
  // and crash upon.

  // Point lexer input at output stream
  smtlib_tokin = in_stream;

  fprintf(out_stream, "(set-logic %s)\n", logic.c_str());
  fprintf(out_stream, "(set-info :status unknown)\n");
  fprintf(out_stream, "(set-option :produce-models true)\n");

  // Fetch solver name and version.
  fprintf(out_stream, "(get-info :name)\n");
  fflush(out_stream);
  smtlib_send_start_code = 1;
  smtlibparse(TOK_START_INFO);

  // As a result we should have a single entry in a list of sexprs.
  class sexpr *sexpr = smtlib_output;
  assert(sexpr->sexpr_list.size() == 1 &&
         "More than one sexpr response to get-info name");
  class sexpr &s = sexpr->sexpr_list.front();

  // Should have a keyword followed by a string?
  assert(s.token == 0 && s.sexpr_list.size() == 2 && "Bad solver name format");
  class sexpr &keyword = s.sexpr_list.front();
  class sexpr &value = s.sexpr_list.back();
  assert(keyword.token == TOK_KEYWORD && keyword.data == ":name" &&
         "Bad get-info :name response from solver");
  assert(value.token == TOK_STRINGLIT && "Non-string solver name response");
  solver_name = value.data;
  delete smtlib_output;

  // Duplicate / boilerplate;
  fprintf(out_stream, "(get-info :version)\n");
  fflush(out_stream);
  smtlib_send_start_code = 1;
  smtlibparse(TOK_START_INFO);

  sexpr = smtlib_output;
  assert(sexpr->sexpr_list.size() == 1 &&
         "More than one sexpr response to get-info version");
  class sexpr &v = sexpr->sexpr_list.front();

  assert(v.token == 0 && v.sexpr_list.size() == 2 && "Bad solver version fmt");
  class sexpr &kw = v.sexpr_list.front();
  class sexpr &val = v.sexpr_list.back();
  assert(kw.token == TOK_KEYWORD && kw.data == ":version" &&
         "Bad get-info :version response from solver");
  assert(val.token == TOK_STRINGLIT && "Non-string solver version response");
  solver_version = val.data;
  delete smtlib_output;
}

smtlib_convt::~smtlib_convt()
{
  delete_all_asts();
}

std::string
smtlib_convt::sort_to_string(const smt_sort *s) const
{
  const smtlib_smt_sort *sort = static_cast<const smtlib_smt_sort *>(s);
  std::stringstream ss;

  switch (sort->id) {
  case SMT_SORT_INT:
    return "Int";
  case SMT_SORT_REAL:
    return "Real";
  case SMT_SORT_BV:
    ss << "(_ BitVec " << sort->data_width << ")";
    return ss.str();
  case SMT_SORT_ARRAY:
    ss << "(Array " << sort_to_string(sort->domain) << " "
                    << sort_to_string(sort->range) << ")";
    return ss.str();
  case SMT_SORT_BOOL:
    return "Bool";
  case SMT_SORT_STRUCT:
  case SMT_SORT_UNION:
  default:
    std::cerr << "Unexpected sort in smtlib_convt" << std::endl;
    abort();
  }
}

unsigned int
smtlib_convt::emit_terminal_ast(const smtlib_smt_ast *ast, std::string &output)
{
  std::stringstream ss;
  const smtlib_smt_sort *sort = static_cast<const smtlib_smt_sort *>(ast->sort);

  switch (ast->kind) {
  case SMT_FUNC_INT:
    // Just the literal number itself.
    output = integer2string(ast->intval);
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
    // Irritatingly, the number may be higher than the actual bitwidth permits.
    assert(sort->data_width <= 64 && "smtlib printer assumes no numbers more "
           "than 64 bits wide, sorry");
    uint64_t theval = ast->intval.to_int64();
    if (sort->data_width < 64) {
      uint64_t mask = 1ULL << sort->data_width;
      mask -= 1;
      theval &= mask;
    }
    assert(sort->data_width != 0);
    ss << "(_ bv" << theval << " " << sort->data_width << ")";
    output = ss.str();
    return 0;
  }
  case SMT_FUNC_REAL:
    // Give up
    ss << ast->realval;
    output = ss.str();
    return 0;
  case SMT_FUNC_SYMBOL:
    // All symbols to be emitted braced within |'s
    ss << "|" << ast->symname << "|";
    output = ss.str();
    return 0;
  default:
    std::cerr << "Invalid terminal AST kind" << std::endl;
    abort();
  }
}

unsigned int
smtlib_convt::emit_ast(const smtlib_smt_ast *ast, std::string &output)
{
  unsigned int brace_level = 0, i;
  std::string args[4];

  switch (ast->kind) {
  case SMT_FUNC_HACKS:
  case SMT_FUNC_INVALID:
    std::cerr << "Invalid SMT function application reached SMTLIB printer"
              << std::endl;
    abort();
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

  for (i = 0; i < ast->num_args; i++)
    brace_level += emit_ast(static_cast<const smtlib_smt_ast *>(ast->args[i]),
                            args[i]);

  // Get a temporary sym name
  unsigned int tempnum = temp_sym_count.back()++;
  std::stringstream ss;
  ss << temp_prefix << tempnum;
  std::string tempname = ss.str();

  // Emit a let, assigning the result of this AST func to the sym.
  // For some reason let requires a double-braced operand.
  fprintf(out_stream, "(let ((%s (", tempname.c_str());

  // This asts function
  assert((int)ast->kind <= (int)expr2t::end_expr_id);
  if (ast->kind == SMT_FUNC_EXTRACT) {
    // Extract is an indexed function
    fprintf(out_stream, "(_ extract %d %d)", ast->extract_high,
                                             ast->extract_low);
  } else {
    fprintf(out_stream, "%s", smt_func_name_table[ast->kind].c_str());
  }

  // Its operands
  for (i = 0; i < ast->num_args; i++)
    fprintf(out_stream, " %s", args[i].c_str());

  // End func enclosing brace, then operand to let (two braces).
  fprintf(out_stream, ")))\n");

  // We end with one additional brace level.
  output = tempname;
  return brace_level + 1;
}

smt_convt::resultt
smtlib_convt::dec_solve()
{
  pre_solve();

  // Set some preliminaries, logic and so forth.
  // Declare all the symbols + sorts
  // Emit constraints
  // check-sat

  fprintf(out_stream, "(check-sat)\n");

  // Flush out command, starting model check
  fflush(out_stream);

  // If we're just outputing to a file, this is where we terminate.
  if (in_stream == NULL)
    return smt_convt::P_SMTLIB;

  // And read in the output
  smtlib_send_start_code = 1;
  smtlibparse(TOK_START_SAT);

  // This should generate on sexpr. See what it is.
  if (smtlib_output->token == TOK_KW_SAT) {
    return smt_convt::P_SATISFIABLE;
  } else if (smtlib_output->token == TOK_KW_UNSAT) {
    return smt_convt::P_UNSATISFIABLE;
  } else if (smtlib_output->token == TOK_KW_ERROR) {
    std::cerr << "SMTLIB solver returned error: \"" << smtlib_output->data
              << "\"" << std::endl;
    return smt_convt::P_ERROR;
  } else {
    std::cerr << "Unrecognized check-sat output from smtlib solver"
              << std::endl;
    abort();
  }
}

expr2tc
smtlib_convt::get_bv(const type2tc &t, smt_astt a)
{

  // This should always be a symbol.
  const smtlib_smt_ast *sa = static_cast<const smtlib_smt_ast*>(a);
  assert(sa->kind == SMT_FUNC_SYMBOL && "Non-symbol in smtlib expr get_bv()");
  std::string name = sa->symname;

  fprintf(out_stream, "(get-value (|%s|))\n", name.c_str());
  fflush(out_stream);
  smtlib_send_start_code = 1;
  smtlibparse(TOK_START_VALUE);

  if (smtlib_output->token == TOK_KW_ERROR) {
    std::cerr << "Error from smtlib solver when fetching literal value: \""
              << smtlib_output->data << "\"" << std::endl;
    abort();
  } else if (smtlib_output->token != 0) {
    std::cerr << "Unrecognized response to get-value from smtlib solver"
              << std::endl;
  }

  // Unpack our value from response list.
  assert(smtlib_output->sexpr_list.size() == 1 && "More than one response to "
         "get-value from smtlib solver");
  sexpr &response = *smtlib_output->sexpr_list.begin();
  // Now we have a valuation pair. First is the symbol
  assert(response.sexpr_list.size() == 2 && "Expected 2 operands in "
         "valuation_pair_list from smtlib solver");
  std::list<sexpr>::iterator it = response.sexpr_list.begin();
  sexpr &symname = *it++;
  sexpr &respval = *it++;
  assert(symname.token == TOK_SIMPLESYM && symname.data == name &&
         "smtlib solver returned different symbol from get-value");

  // Attempt to read an integer.
  BigInt m;
  bool was_integer = true;
  if (respval.token == TOK_DECIMAL) {
    m = string2integer(respval.data);
  } else if (respval.token == TOK_NUMERAL) {
    std::cerr << "Numeral value for integer symbol from smtlib solver"
              << std::endl;
    abort();
  } else if (respval.token == TOK_HEXNUM) {
    std::string data = respval.data.substr(2);
    m = string2integer(data, 16);
  } else if (respval.token == TOK_BINNUM) {
    std::string data = respval.data.substr(2);
    m = string2integer(data, 2);
  } else {
    was_integer = false;
  }

  // Generate the appropriate expr.
  expr2tc result;
  if (is_bv_type(t)) {
    assert(was_integer && "smtlib solver didn't provide integer response to "
           "integer get-value");
    result = constant_int2tc(t, m);
  } else if (is_fixedbv_type(t)) {
    assert(!int_encoding && "Can't parse reals right now in smtlib solver "
           "responses");
    assert(was_integer && "smtlib solver didn't provide integer/bv response to "
           "fixedbv get-value");
    const fixedbv_type2t &fbtype = to_fixedbv_type(t);
    fixedbv_spect spec(fbtype.width, fbtype.integer_bits);
    fixedbvt fbt;
    fbt.spec = spec;
    fbt.from_integer(m);
    result = constant_fixedbv2tc(fbt);
  } else if (is_bool_type(t)) {
    if (respval.token == TOK_KW_TRUE) {
      result = constant_bool2tc(true);
    } else if (respval.token == TOK_KW_FALSE) {
      result = constant_bool2tc(false);
    } else {
      std::cerr << "Unexpected token reading value of boolean symbol from "
                   "smtlib solver" << std::endl;
    }
  } else {
    abort();
  }

  delete smtlib_output;
  return result;
}

expr2tc
smtlib_convt::get_array_elem (const smt_ast *array, uint64_t index,
    const type2tc &t)
{

  // This should always be a symbol.
  const smtlib_smt_ast *sa = static_cast<const smtlib_smt_ast*>(array);
  assert(sa->kind == SMT_FUNC_SYMBOL && "Non-symbol in smtlib get_array_elem");
  std::string name = sa->symname;

  // XXX -- double bracing this may be a Z3 ecentricity
  unsigned long domain_width = array->sort->domain_width;
  fprintf(out_stream,
      "(get-value ((select |%s| (_ bv%" PRIu64 " %" PRIu64 "))))\n",
      name.c_str(), index, domain_width);
  fflush(out_stream);
  smtlib_send_start_code = 1;
  smtlibparse(TOK_START_VALUE);

  if (smtlib_output->token == TOK_KW_ERROR) {
    std::cerr << "Error from smtlib solver when fetching literal value: \""
              << smtlib_output->data << "\"" << std::endl;
    abort();
  } else if (smtlib_output->token != 0) {
    std::cerr << "Unrecognized response to get-value from smtlib solver"
              << std::endl;
  }

  // Unpack our value from response list.
  assert(smtlib_output->sexpr_list.size() == 1 && "More than one response to "
         "get-value from smtlib solver");
  sexpr &response = *smtlib_output->sexpr_list.begin();
  // Now we have a valuation pair. First is the symbol
  assert(response.sexpr_list.size() == 2 && "Expected 2 operands in "
         "valuation_pair_list from smtlib solver");
  std::list<sexpr>::iterator it = response.sexpr_list.begin();
  it++; // Echo of what we selected
  sexpr &respval = *it++;

  // Attempt to read an integer.
  BigInt m;
  bool was_integer = true;
  if (respval.token == TOK_DECIMAL) {
    m = string2integer(respval.data);
  } else if (respval.token == TOK_NUMERAL) {
    std::cerr << "Numeral value for integer symbol from smtlib solver"
              << std::endl;
    abort();
  } else if (respval.token == TOK_HEXNUM) {
    std::string data = respval.data.substr(2);
    m = string2integer(data, 16);
  } else if (respval.token == TOK_BINNUM) {
    std::string data = respval.data.substr(2);
    m = string2integer(data, 2);
  } else {
    was_integer = false;
  }

  // Generate the appropriate expr.
  expr2tc result;
  if (is_bv_type(t)) {
    assert(was_integer && "smtlib solver didn't provide integer response to "
           "integer get-value");
    result = constant_int2tc(t, m);
  } else if (is_fixedbv_type(t)) {
    assert(!int_encoding && "Can't parse reals right now in smtlib solver "
           "responses");
    assert(was_integer && "smtlib solver didn't provide integer/bv response to "
           "fixedbv get-value");
    const fixedbv_type2t &fbtype = to_fixedbv_type(t);
    fixedbv_spect spec(fbtype.width, fbtype.integer_bits);
    fixedbvt fbt;
    fbt.spec = spec;
    fbt.from_integer(m);
    result = constant_fixedbv2tc(fbt);
  } else if (is_bool_type(t)) {
    if (respval.token == TOK_KW_TRUE) {
      result = constant_bool2tc(true);
    } else if (respval.token == TOK_KW_FALSE) {
      result = constant_bool2tc(false);
    } else if (respval.token == TOK_BINNUM) {
      assert(respval.data.size() == 3 && "Boolean-typed binary number should "
          "be 3 characters long (e.g. #b0)");

      std::string data = respval.data.substr(2);
      if (data[0] == '0')
        result = gen_false_expr();
      else if (data[0] == '1')
        result = gen_true_expr();
      else {
        std::cerr << "Unrecognized boolean-typed binary number format";
        std::cerr << std::endl;
        abort();
      }
    } else {
      std::cerr << "Unexpected token reading value of boolean symbol from "
                   "smtlib solver" << std::endl;
    }
  } else {
    abort();
  }

  delete smtlib_output;
  return result;
}


expr2tc
smtlib_convt::get_bool(smt_astt a)
{
  tvt res = l_get(a);
  if (res.is_true())
    return gen_true_expr();
  else if (res.is_false())
    return gen_false_expr();
  else {
    std::cerr << "Non-true, non-false value read from smtlib model" <<std::endl;
    abort();
  }
}

tvt
smtlib_convt::l_get(const smt_ast *a)
{
  fprintf(out_stream, "(get-value (");

  std::string output;
  unsigned int brace_level =
    emit_ast(static_cast<const smtlib_smt_ast*>(a), output);
  fprintf(out_stream, "%s", output.c_str());

  // Emit a ton of end braces.
  for (unsigned int i = 0; i < brace_level; i++)
    fputc(')', out_stream);

  fprintf(out_stream, "))\n");

  fflush(out_stream);
  smtlib_send_start_code = 1;
  smtlibparse(TOK_START_VALUE);

  if (smtlib_output->token == TOK_KW_ERROR) {
    std::cerr << "Error from smtlib solver when fetching literal value: \""
              << smtlib_output->data << "\"" << std::endl;
    abort();
  } else if (smtlib_output->token != 0) {
    std::cerr << "Unrecognized response to get-value from smtlib solver"
              << std::endl;
  }

  // First layer: valuation pair list. Should have one item.
  assert(smtlib_output->sexpr_list.size() == 1 && "Unexpected number of "
         "responses to get-value from smtlib solver");
  sexpr &pair = *smtlib_output->sexpr_list.begin();
  // Should have two entries
  assert(pair.sexpr_list.size() == 2 && "Valuation pair in smtlib get-value "
         "output without two operands");
  std::list<sexpr>::const_iterator it = pair.sexpr_list.begin();
  const sexpr &first = *it++;
  (void)first;
  const sexpr &second = *it++;
//  assert(first.token == TOK_SIMPLESYM && first.data == ss.str() &&
//         "Unexpected valuation variable from smtlib solver");

  // And finally we have our value. It should be true or false.
  tvt result;
  if (second.token == TOK_KW_TRUE) {
    result = tvt(true);
  } else if (second.token == TOK_KW_FALSE) {
    result = tvt(false);
  } else {
    std::cerr << "Unexpected literal valuation from smtlib solver" << std::endl;
    abort();
  }

  delete smtlib_output;
  return result;
}

const std::string
smtlib_convt::solver_text()
{
  if (in_stream == NULL) {
    // Text output
    return solver_name;
  }

  return solver_name + " version " + solver_version;
}

void
smtlib_convt::assert_ast(const smt_ast *a)
{
  const smtlib_smt_ast *sa = static_cast<const smtlib_smt_ast *>(a);

  // Encode an assertion
  fprintf(out_stream, "(assert\n");

  // The algorithm: descend through the AST operands, binding values to
  // temporary symbols, then emit functions on those temporary symbols.
  // All recursively. The non-trivial bit is tracking how many ending
  // braces are required.
  // This is inspired by the output from Z3 that I've seen.
  std::string output;
  unsigned int brace_level = emit_ast(sa, output);

  // Emit the final temporary symbol - this is what gets asserted.
  fprintf(out_stream, "%s", output.c_str());

  // Emit a ton of end braces.
  for (unsigned int i = 0; i < brace_level; i++)
    fputc(')', out_stream);

  // Final brace for closing the 'assert'.
  fprintf(out_stream, ")\n");
}

smt_ast *
smtlib_convt::mk_func_app(const smt_sort *s, smt_func_kind k,
                          const smt_ast * const *args,
                          unsigned int numargs)
{
  assert(numargs <= 4 && "Too many arguments to smtlib mk_func_app");
  smtlib_smt_ast *a = new smtlib_smt_ast(this, s, k);
  a->num_args = numargs;
  for (unsigned int i = 0; i < 4; i++)
    a->args[i] = args[i];

  return a;
}

smt_sort *
smtlib_convt::mk_sort(const smt_sort_kind k __attribute__((unused)), ...)
{
  va_list ap;
  smtlib_smt_sort *s = NULL, *dom, *range;
  unsigned long uint;
  int thebool;

  va_start(ap, k);
  switch (k) {
  case SMT_SORT_INT:
    thebool = va_arg(ap, int);
    s = new smtlib_smt_sort(k, thebool);
    break;
  case SMT_SORT_REAL:
    s = new smtlib_smt_sort(k);
    break;
  case SMT_SORT_BV:
    uint = va_arg(ap, unsigned long);
    thebool = va_arg(ap, int);
    assert(uint != 0);
    s = new smtlib_smt_sort(k, uint);
    break;
  case SMT_SORT_ARRAY:
    dom = va_arg(ap, smtlib_smt_sort *); // Consider constness?
    range = va_arg(ap, smtlib_smt_sort *);
    s = new smtlib_smt_sort(k, dom, range);
    break;
  case SMT_SORT_BOOL:
    s = new smtlib_smt_sort(k);
    break;
  default:
    assert(0);
  }

  return s;
}

smt_ast *
smtlib_convt::mk_smt_int(const mp_integer &theint, bool sign)
{
  smt_sort *s = mk_sort(SMT_SORT_INT, sign);
  smtlib_smt_ast *a = new smtlib_smt_ast(this, s, SMT_FUNC_INT);
  a->intval = theint;
  return a;
}

smt_ast *
smtlib_convt::mk_smt_real(const std::string &str)
{
  smt_sort *s = mk_sort(SMT_SORT_REAL);
  smtlib_smt_ast *a = new smtlib_smt_ast(this, s, SMT_FUNC_REAL);
  a->realval = str;
  return a;
}

smt_ast *
smtlib_convt::mk_smt_bvint(const mp_integer &theint, bool sign, unsigned int w)
{
  smt_sort *s = mk_sort(SMT_SORT_BV, w, sign);
  smtlib_smt_ast *a = new smtlib_smt_ast(this, s, SMT_FUNC_BVINT);
  a->intval = theint;
  return a;
}

smt_ast *
smtlib_convt::mk_smt_bvfloat(const ieee_floatt &thereal,
                             unsigned ew, unsigned sw)
{
  std::cerr << "Can't create floating points on smtlib yet" << std::endl;
  abort();
}

smt_astt smtlib_convt::mk_smt_bvfloat_nan(unsigned ew, unsigned sw)
{
  std::cerr << "Can't create NaNs on smtlib yet" << std::endl;
  abort();
}

smt_astt smtlib_convt::mk_smt_bvfloat_inf(bool sgn, unsigned ew, unsigned sw)
{
  std::cerr << "Can't create Infs on smtlib yet" << std::endl;
  abort();
}

smt_astt smtlib_convt::mk_smt_bvfloat_rm(ieee_floatt::rounding_modet rm)
{
  std::cerr << "Can't create rounding modes on smtlib yet" << std::endl;
  abort();
}

smt_astt smtlib_convt::mk_smt_typecast_from_bvfloat(const typecast2t& cast)
{
  std::cerr << "Can't cast floating point on smtlib yet" << std::endl;
  abort();
}

smt_astt smtlib_convt::mk_smt_typecast_to_bvfloat(const typecast2t& cast)
{
  std::cerr << "Can't cast floating point on smtlib yet" << std::endl;
  abort();
}

smt_astt smtlib_convt::mk_smt_bvfloat_arith_ops(const expr2tc& expr)
{
  std::cerr << "Can't create floating point arith op on smtlib yet" << std::endl;
  abort();
}

smt_astt smtlib_convt::mk_smt_nearbyint_from_float(const nearbyint2t& expr)
{
  std::cerr << "Can't create floating point nearbyint expression on smtlibt yet" << std::endl;
  abort();
}

smt_ast *
smtlib_convt::mk_smt_bool(bool val)
{
  smtlib_smt_ast *a =
    new smtlib_smt_ast(this,mk_sort(SMT_SORT_BOOL), SMT_FUNC_BOOL);
  a->boolval = val;
  return a;
}

smt_ast *
smtlib_convt::mk_array_symbol(const std::string &name, const smt_sort *s,
                              smt_sortt array_subtype __attribute__((unused)))
{
  return mk_smt_symbol(name, s);
}

smt_ast *
smtlib_convt::mk_smt_symbol(const std::string &name, const smt_sort *s)
{
  smtlib_smt_ast *a = new smtlib_smt_ast(this, s, SMT_FUNC_SYMBOL);
  a->symname = name;

  symbol_tablet::iterator it = symbol_table.find(name);

  if (it != symbol_table.end())
    return a;

  // Record the type of this symbol
  struct symbol_table_rec record = { name, ctx_level, s };
  symbol_table.insert(record);

  if (s->id == SMT_SORT_STRUCT || s->id == SMT_SORT_UNION)
    return a;;

  // As this is the first time, declare that symbol to the solver.
  fprintf(out_stream, "(declare-fun |%s| () %s)\n", name.c_str(),
         sort_to_string(s).c_str());

  return a;
}

smt_sort *
smtlib_convt::mk_struct_sort(const type2tc &type __attribute__((unused)))
{
  std::cerr << "Attempted to make struct type in smtlib conversion" <<std::endl;
  abort();
}

smt_ast *
smtlib_convt::mk_extract(const smt_ast *a, unsigned int high, unsigned int low,
                         const smt_sort *s)
{
  smtlib_smt_ast *n = new smtlib_smt_ast(this, s, SMT_FUNC_EXTRACT);
  n->extract_high = high;
  n->extract_low = low;
  n->num_args = 1;
  n->args[0] = a;
  return n;
}

int
smtliberror(int startsym __attribute__((unused)), const std::string &error)
{
  std::cerr << "SMTLIB response parsing error: \"" << error << "\""
            << std::endl;
  abort();
}

void
smtlib_convt::push_ctx()
{
  smt_convt::push_ctx();
  temp_sym_count.push_back(temp_sym_count.back());

  fprintf(out_stream, "(push 1)\n");
}

void
smtlib_convt::pop_ctx()
{
  fprintf(out_stream, "(pop 1)\n");

  // Wipe this level of symbol table.
  symbol_tablet::nth_index<1>::type &syms_numindex = symbol_table.get<1>();
  syms_numindex.erase(ctx_level);
  temp_sym_count.pop_back();

  smt_convt::pop_ctx();
}

const smt_ast *
smtlib_convt::convert_array_of(smt_astt init_val, unsigned long domain_width)
{
  return default_convert_array_of(init_val, domain_width, this);
}

void
smtlib_convt::add_array_constraints_for_solving()
{
  // None required
  return;
}

void
smtlib_convt::push_array_ctx(void)
{
  return;
}

void
smtlib_convt::pop_array_ctx(void)
{
  return;
}

const std::string smtlib_convt::temp_prefix = "?x";
