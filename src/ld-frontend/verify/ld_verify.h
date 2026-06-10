#pragma once

#include <string>

// ld_verify orchestrates the full pipeline:
//   PLCopen XML → Parser → TypeChecker → LdIRBuilder → ld_converter
//   → property_encoder → ESBMC engine → JSON report
//
// This class is used by the ld-verify CLI and by ld_languaget.
struct LdVerifyOptions
{
  std::string program_path; // PLCopen XML or .ld input
  std::string props_path;   // YAML property file (optional)
  std::string strategy;     // "k-induction" | "bmc"
  unsigned bmc_unwind = 100;
  bool fault_injection = false;
};

struct LdVerifyResult
{
  enum class Verdict
  {
    Safe,
    Violation,
    Unknown,
    Incomplete,
    Error
  };

  Verdict verdict = Verdict::Unknown;
  std::string property_id;
  std::string description;
  std::string raw_output; // captured ESBMC output

  // Serialise to JSON string
  std::string to_json() const;
};

class LdVerifyRunner
{
public:
  // Run the full pipeline.  The context is owned by the caller.
  LdVerifyResult run(const LdVerifyOptions &opts);
};
