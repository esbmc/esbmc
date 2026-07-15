// Mode C (C-Live) reachability harness for esbmc/esbmc#5905.
//
// Production site: src/esbmc/parseoptions/goto_program.cpp,
// esbmc_parseoptionst::create_goto_program(), added branch:
//
//   if (cmdline.args.empty() && cmdline.isset("sol"))
//     cmdline.args.push_back(cmdline.getval("sol"));
//
// FULL-FIDELITY HARNESS DECLARED INFEASIBLE: the real cmdlinet type
// (src/util/cmdline.h) is defined in terms of
// boost::program_options::variables_map, and <boost/program_options.hpp>
// does not parse under ESBMC's bundled C++ STL model (probed directly;
// see report -- errors on std::basic_string::size_type, non-const
// std::map::find, std::wstring, and missing <iosfwd>). create_goto_program
// itself additionally requires a fully constructed esbmc_parseoptionst
// (language_uit + parseoptions_baset), a real optionst, and a real
// goto_functionst, none of which is a small dependency surface. This
// harness therefore extracts the branch's condition/effect using the real
// field type for `args` (std::vector<std::string>, matching
// cmdlinet::args) and a pure-nondet G4 stub for the two boost-backed
// primitives, isset()/getval().
//
// The precondition below is not an arbitrary "assume the branch true":
// it is grounded in the real command-line grammar, established by
// reading (not executing) the production sources:
//   - src/esbmc/options.cpp:239-241 registers "sol" as an independent
//     named value-option ({"sol", value<string>(), ...}).
//   - src/util/cmdline.cpp:281 binds the *only* positional slot to
//     "input-file" (`p.add("input-file", -1)`), not "sol".
//   - src/util/cmdline.cpp:166 clears `args`; cmdline.cpp:334-335 only
//     repopulate it from vm["input-file"], i.e. from a positional token.
// Hence "esbmc --sol foo.solast" (no positional token) is a real
// invocation for which isset("sol")==true while args stays empty --
// confirmed independently by an end-to-end run of the patched binary
// (see report).

#include <vector>
#include <string>

extern "C" bool nondet_bool();

int main()
{
  std::vector<std::string> args; // matches cmdlinet::args post cmdline.cpp:166 clear()

  bool isset_sol = nondet_bool();
  // G3: isset("sol") can be true while args is empty -- see file header citation.
  __ESBMC_assume(isset_sol);

  if (args.empty() && isset_sol)
  {
    __ESBMC_unreachable();
    args.push_back("stand-in-getval-sol-value"); // getval() stubbed: content irrelevant to reachability
  }

  return 0;
}
