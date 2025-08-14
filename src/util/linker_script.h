// Applies GNU linker script

#pragma once

#include <memory>
#include <goto-programs/goto_functions.h>

class linker_script
{
 public:
  // e.g.: FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 512K;
  struct memory_region_t {
    unsigned origin = 0;  // ORIGIN
    unsigned length = 0;  // LENGTH in bytes
    enum perms_t
    {
      RX,
      R,
      RW,
      RWX
    } perms = RWX;
    exprt to_range_assumption(const symbolt &sym) const;
  };
  exprt no_overlapping_symbols(const symbolt &sym) const;

  // e.g. .fcu 0x20001000 ALIGN(8) : { *(.fcu) } > SRAM
  struct output_section_t {
    ;                // .text, .data, .fcu
    std::optional<uint64_t> addr;    // Optional fixed start address: ".fcu 0x20001000 : { ... }"
    std::optional<uint64_t> align;   // ALIGN(n) at section start (subset)
    std::vector<std::string> inputs; // Which input sections map here
    std::shared_ptr<memory_region_t> memory_region; // "> FLASH"
    std::vector<symbolt> symbols;
    exprt to_ordering_assumption() const;
  };

  struct script_t {
    std::unordered_map<std::string, std::shared_ptr<memory_region_t>> memory;
    std::unordered_map<std::string, std::shared_ptr<output_section_t>> sections;
  };
    
 explicit linker_script(const std::string &ld_script_src);
 std::optional<script_t> parse(const std::string &ld_script_src);



 bool apply(contextt &program);
 void generate_example();

private:
  script_t layout;

};
