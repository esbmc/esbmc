#include <ld-frontend/ld_language.h>
#include <ld-frontend/verify/ld_verify.h>
#include <iostream>
#include <string>

static void print_usage()
{
  std::cerr <<
    "Usage: ld-verify [options] <program.xml>\n"
    "\n"
    "Options:\n"
    "  --props <file.yaml>   YAML property specification\n"
    "  --strategy <s>        k-induction | bmc | portfolio (default: k-induction)\n"
    "  --unwind <n>          BMC unwind bound (default: 100)\n"
    "  --fault-injection     Enable fault-injection mode (for WP1 validation)\n"
    "  --show-parse          Print the parsed LD program and exit\n"
    "  --help                Print this message\n"
    "\n"
    "ld-verify checks safety properties of IEC 61131-3 Ladder Diagram programs\n"
    "exported in PLCopen XML format.  Results are reported as JSON to stdout.\n";
}

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    print_usage();
    return 1;
  }

  LdVerifyOptions opts;
  opts.strategy = "k-induction";
  bool show_parse = false;

  for (int i = 1; i < argc; ++i)
  {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h")
    {
      print_usage();
      return 0;
    }
    else if (arg == "--props" && i + 1 < argc)
      opts.props_path = argv[++i];
    else if (arg == "--strategy" && i + 1 < argc)
      opts.strategy = argv[++i];
    else if (arg == "--unwind" && i + 1 < argc)
    {
      try
      {
        opts.bmc_unwind = static_cast<unsigned>(std::stoul(argv[++i]));
      }
      catch (const std::exception &)
      {
        std::cerr << "error: --unwind requires a non-negative integer\n";
        return 1;
      }
    }
    else if (arg == "--fault-injection")
      opts.fault_injection = true;
    else if (arg == "--show-parse")
      show_parse = true;
    else if (arg[0] != '-')
      opts.program_path = arg;
    else
    {
      std::cerr << "Unknown option: " << arg << "\n";
      print_usage();
      return 1;
    }
  }

  if (opts.program_path.empty())
  {
    std::cerr << "Error: no input file specified\n";
    print_usage();
    return 1;
  }

  // Use ld_languaget directly for parse + show_parse without the full ESBMC driver.
  if (show_parse)
  {
    ld_languaget lang;
    if (!opts.props_path.empty())
      lang.set_props_path(opts.props_path);
    if (lang.parse(opts.program_path))
    {
      std::cerr << "Parse failed.\n";
      return 1;
    }
    lang.show_parse(std::cout);
    return 0;
  }

  // Full verification: delegate to LdVerifyRunner which invokes ESBMC.
  LdVerifyRunner runner;
  LdVerifyResult result = runner.run(opts);
  std::cout << result.to_json();
  switch (result.verdict)
  {
  case LdVerifyResult::Verdict::Safe:
    return 0;
  case LdVerifyResult::Verdict::Violation:
    return 10;
  case LdVerifyResult::Verdict::Unknown:
  case LdVerifyResult::Verdict::Incomplete:
    return 1;
  case LdVerifyResult::Verdict::Error:
    return 2;
  }
  return 2;
}
