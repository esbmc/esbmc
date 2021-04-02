/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <util/cmdline.h>

std::string verification_file;

cmdlinet::~cmdlinet()
{
  clear();
}

void cmdlinet::clear()
{
  vm.clear();
  args.clear();
}

bool cmdlinet::isset(const char *option) const
{
  //std::cout<<"calling isset for "<<option<<" with value "<<vm.count(option)<<"\n";
  return vm.count(option);
}

const char *cmdlinet::getval(char option) const
{
  return getval(option + "");
}

const std::list<std::string> &cmdlinet::get_values(const char *option) const
{
  auto src = vm[option].as<std::vector<std::string>>();
  //  std::cout<<"Values for "<<option<<" are: \n"<<src<<"\n";
  auto dest = new std::list<std::string>(src.begin(), src.end());
  return *dest;
}

const char *cmdlinet::getval(const char *option) const
{
  if(!vm.count(option))
    return (const char *)nullptr;
  auto &value = vm[option].value();
  if(auto v = boost::any_cast<int>(&value))
  {
    auto res = std::to_string(*v).c_str();
    char *newstr = new char[strlen(res) + 1];
    strcpy(newstr, res);
    return newstr;
  }
  else if(auto v = boost::any_cast<std::string>(&value))
    return v->c_str();
  else if(auto v = boost::any_cast<std::vector<std::string>>(&value))
    return v->front().c_str();

  else
    return (const char *)nullptr;
}

void cmdlinet::parse(int argc, const char **argv)
{
  clear();

  po::options_description group0("file_name");
  group0.add_options()(
    "input-file",
    po::value<std::vector<std::string>>()->value_name("file.c ..."),
    "source file names");
  po::positional_options_description p;
  p.add("input-file", -1);

  po::options_description group1("Options");
  group1.add_options()("h", "show help")("help,?", "show help");
  po::options_description group2("Printing options");
  group2.add_options()("symbol-table-only", "only show symbol table")(
    "symbol-table-too",
    "show symbol table and verify")("parse-tree-only", "only show parse tree")(
    "parse-tree-too", "show parse tree and verify")(
    "goto-functions-only", "only show goto program")(
    "goto-functions-too", "show goto program and verify")(
    "program-only", "only show program expression")(
    "program-too", "show program expression and verify")(
    "ssa-symbol-table", "show symbol table along with SSA")("ssa-guards", "")(
    "ssa-sliced", "print the sliced SSAs")("ssa-no-location", "")(
    "smt-formula-only", "only show SMT formula (not supported by all solvers)")(
    "smt-formula-too ",
    "show SMT formula (not supported by all solvers) and verify")(
    "smt-model",
    "show SMT model (not supported by all solvers), if the formula is SAT");
  po::options_description group3("Trace");
  group3.add_options()(
    "quiet", " do not print unwinding information during symbolic execution")(
    "symex-trace", "print instructions during symbolic execution")(
    "ssa-trace", "print SSA during SMT encoding")(
    "ssa-smt-trace", "print generated SMT during SMT encoding")(
    "symex-ssa-trace", "print generated SSA during symbolic execution")(
    "show-goto-value-sets ", "show value-set analysis for the goto functions")(
    "show-symex-value-sets",
    "show value-set analysis during symbolic execution");
  po::options_description group4("Frontend");
  group4.add_options()(
    "include,I",
    po::value<std::vector<std::string>>()->value_name("path"),
    "set include path")(
    "define,D",
    po::value<std::vector<std::string>>()->value_name("macro"),
    "define preprocessor macro")(
    "warning,W", po::value<std::vector<std::string>>(), "")(
    "force,f", po::value<std::vector<std::string>>(), "")(
    "preprocess", "stop after preprocessing")(
    "no-inlining", "disable inlining function calls")(
    "full-inlining",
    "perform full inlining of function calls")("all-claims", "keep all claims")(
    "show-loops",
    "show the loops in the program")("show-claims", "only show claims")(
    "show-vcc", "show the verification conditions")(
    "document-subgoals", "generate subgoals documentation")(
    "no-arch", "don't set up an architecture")(
    "no-library", "disable built-in abstract C library")(
    "binary", "read goto program instead of source code")(
    "little-endian", "allow little-endian word-byte conversions")(
    "big-endian", "allow big-endian word-byte conversions")(
    "16", "set width of machine word (default is 64)")(
    "32", "set width of machine word (default is 64)")(
    "64", "set width of machine word (default is 64)")(
    "version", "show current ESBMC version and exit")(
    "witness-output",
    po::value<std::string>(),
    "generate the verification result witness in GraphML format")(
    "witness-producer", po::value<std::string>(), "")(
    "witness-programfile", po::value<std::string>(), "")(
    "old-frontend", "parse source files using our old frontend (deprecated)")(
    "result-only", "do not print the counter-example")(
    "i386-linux", "set Linux/I386 architecture (default)")(
    "i386-macos", "set MACOS/I386 architecture")(
    "i386-win32",
    "set Windows/I386 architecture")("ppc-macos", "set PPC/I386 architecture")(
    "funsigned-char", "make \"char\" unsigned by default")(
    "fms-extensions", "enable microsoft C extensions");
  po::options_description group5("BMC");
  group5.add_options()(
    "function",
    po::value<std::string>()->value_name("name"),
    "set main function name")(
    "claim", po::value<int>()->value_name("nr"), "only check specific claim")(
    "depth", po::value<int>()->value_name("nr"), "limit search depth")(
    "unwind", po::value<int>()->value_name("nr"), "unwind nr times")(
    "unwindset",
    po::value<std::string>()->value_name("nr"),
    "unwind given loop nr times")(
    "no-unwinding-assertions", "do not generate unwinding assertions")(
    "partial-loops", "permit paths with partial loops")("unroll-loops", "")(
    "no-slice", "do not remove unused equations")("slice-assumes", "")(
    "extended-try-analysis", "")("skip-bmc", "")("no-return-value-opt", "");
  po::options_description group6("IBMC");
  group6.add_options()(
    "incremental-bmc", "incremental loop unwinding verification")(
    "falsification", "incremental loop unwinding for bug searching")(
    "termination", "incremental loop unwinding assertion verification")(
    "k-step",
    po::value<int>()->value_name("nr"),
    "set k increment (default is 1)")(
    "max-k-step",
    po::value<int>()->value_name("nr"),
    "set max number of iteration (default is 50)")(
    "unlimited-k-steps", "set max number of iteration to UINT_MAX");
  po::options_description group7("Solver");
  group7.add_options()("list-solvers", "list available solvers and exit")(
    "boolector", "use Boolector (default)")("z3", "use Z3")(
    "mathsat", "use MathSAT")("cvc", "use CVC4")("yices", "use Yices")(
    "bv", "use solver with bit-vector arithmetic")(
    "ir",
    "use solver with integer/real arithmetic")("smtlib", "use SMT lib format")(
    "smtlib-solver-prog", po::value<std::string>(), "SMT lib program name")(
    "output",
    po::value<std::string>()->value_name("<filename>"),
    "output VCCs in SMT lib format to given file")(
    "floatbv",
    "encode floating-point using the SMT floating-point theory(default)")(
    "fixedbv", "encode floating-point as fixed bit-vectors")(
    "fp2bv",
    "encode floating-point as bit-vectors(default for solvers that don't "
    "support the SMT floating-point theory)")(
    "tuple-node-flattener", "encode tuples using our tuple to node API")(
    "tuple-sym-flattener", "encode tuples using our tuple to symbol API")(
    "array-flattener", "encode arrays using our array API")(
    "no-return-value-opt",
    "disable return value optimization to compute the stack size");

  po::options_description group8("Incremental SMT");
  group8.add_options()(
    "smt-during-symex", "enable incremental SMT solving (experimental)")(
    "smt-thread-guard",
    "call the solver during thread exploration (experimental)")(
    "smt-symex-guard",
    "call the solver during symbolic execution (experimental)");
  po::options_description group9("Property checking");
  group9.add_options()("no-assertions", "ignore assertions")(
    "no-bounds-check", "do not do array bounds check")(
    "no-div-by-zero-check", "do not do division by zero check")(
    "no-pointer-check", "do not do pointer check")(
    "no-align-check", "do not check pointer alignment")(
    "no-pointer-relation-check", "do not check pointer relations")(
    "nan-check", "check floating-point for NaN")(
    "memory-leak-check", "enable memory leak check")(
    "overflow-check", "enable arithmetic over- and underflow check")(
    "deadlock-check", "enable global and local deadlock check with mutex")(
    "data-races-check", "enable data races check")(
    "lock-order-check", "enable for lock acquisition ordering check")(
    "atomicity-check", "enable atomicity check at visible assignments")(
    "stack-limit",
    po::value<int>()->default_value(-1)->value_name("bytes"),
    "check if stack limit is respected")(
    "error-label",
    po::value<std::string>()->value_name("label"),
    "check if label is unreachable")(
    "force-malloc-success", "do not check for malloc/new failure");
  po::options_description group10("k-induction");
  group10.add_options()("base-case", "check the base case")(
    "forward-condition", "check the forward condition")(
    "inductive-step",
    "check the inductive step")("k-induction", "prove by k-induction ")(
    "k-induction-parallel",
    "prove by k-induction, running each step on a separate process")(
    "k-step",
    po::value<int>()->default_value(1)->value_name("nr"),
    "set k increment (default is 1)")(
    "max-k-step",
    po::value<int>()->default_value(50)->value_name("nr"),
    "set max number of iteration (default is 50)")(
    "show-cex", "print the counter-example produced by the inductive step")(
    "bidirectional",
    "")("unlimited-k-steps", "set max number of iteration to UINT_MAX")(
    "max-inductive-step",
    po::value<int>()->default_value(-1)->value_name("nr"),
    "");
  po::options_description group11("Scheduling");
  group11.add_options()("schedule", "use schedule recording approach")(
    "round-robin", "use the round robin scheduling approach")(
    "time-slice",
    po::value<int>()->default_value(1)->value_name("nr"),
    "set the time slice of the round robin algorithm (default is 1) ");
  po::options_description group12("Concurrency checking");
  group12.add_options()(
    "context-bound",
    po::value<int>()->default_value(-1)->value_name("nr"),
    "limit number of context switches for each thread")(
    "state-hashing", "enable state-hashing, prunes duplicate states")(
    "no-por", "do not do partial order reduction")(
    "all-runs", "check all interleavings, even if a bug was already found");
  po::options_description group13("Miscellaneous options");
  group13.add_options()

    ("memlimit",
     po::value<std::string>()->value_name("limit"),
     "configure memory limit, of form \"100m\" or \"2g\"")(
      "memstats", "print memory usage statistics")(
      "timeout",
      po::value<std::string>()->value_name("t"),
      "configure time limit, integer followed by {s,m,h}")(
      "enable-core-dump", "do not disable core dump output")(
      "no-simplify", "do not simplify any expression")(
      "no-propagation", "disable constant propagation")(
      "interval-analysis",
      "enable interval analysis and add assumes to the program");

  po::options_description group14("DEBUG options");
  group14.add_options()

    // Print commit hash for current binary
    ("git-hash", "")
    // Check if there is two or more assingments to the same SSA instruction
    ("double-assign-check", "")
    // Abort if the program contains a recursion
    ("abort-on-recursion", "")
    // Verbosity of message, probably does nothing
    ("verbosity", po::value<int>(), "")
    // --break-at $insnnum will cause ESBMC to execute a trap
    // instruction when it executes the designated GOTO instruction number.
    ("break-at", po::value<std::string>(), "")
    // I added some intrinsics along the line of "__ESBMC_switch_to_thread"
    // that immediately transitioned to a particular thread and didn't allow
    // any other exploration from that point. Useful for constructing an
    // explicit multithreading path
    ("direct-interleavings", "")
    // I think this dumps the current stack of all threads on an ileave point.
    // Useful for working out the state of _all_ the threads and how they
    // evolve, also see next flag,
    ("print-stack-traces", "")
    // At every ileave point ESBMC stops and asks the user what thread to
    // transition to. Useful again for trying to replicate a particular context
    // switch order, or quickly explore what's reachable.
    ("interactive-ileaves", "");

  cmdline_options.add(group0)
    .add(group1)
    .add(group2)
    .add(group3)
    .add(group4)
    .add(group5)
    .add(group6)
    .add(group7)
    .add(group8)
    .add(group9)
    .add(group10)
    .add(group11)
    .add(group12)
    .add(group13)
    .add(group14);

  store(
    po::command_line_parser(argc, argv)
      .options(cmdline_options)
      .positional(p)
      .run(),
    vm);
  if(vm.count("input-file"))
  {
    args = vm["input-file"].as<std::vector<std::string>>();
    verification_file = args.back();
  }
}
