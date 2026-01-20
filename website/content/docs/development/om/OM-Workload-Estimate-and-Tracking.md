---
title: Operational Model Workload Estimate and Tracking
---


We need to adapt our C++ OMs for the new Clang-based C++ frontend. This page tracks the progress of OM fixes - number of issues/PRs for each OM.  

## Get LOCs for all OMs in descending order: 
In src/cpp/library run the following commands: 
```
find . -name "*" -type f -not -path "./Qt/*" -not -path "./boost/*" -exec wc -l {} \; > file.txt
awk '{print $NF,$0}' file.txt | sort -nr -k2 | cut -d' ' -f2- > output.txt
sed 's/ \+/,/g' output.txt > output.csv
```

## Workload estimate and progress tracking: 
|LOC|OM file name|Status|Remarks|Issues Logged|PRs (fixes)|
|---|---|---|---|---|---|
|2845|./string|$\textcolor{orange}{\textsf{WIP}}$|Parse errors and frontend extensions for more complicated code structures|[Issue832](https://github.com/esbmc/esbmc/issues/832), [Issue991](https://github.com/esbmc/esbmc/issues/991)(2x issues), [Issue989](https://github.com/esbmc/esbmc/issues/989)|[PR988](https://github.com/esbmc/esbmc/pull/988), [PR1180](https://github.com/esbmc/esbmc/pull/1180), various PRs to resolve the umbrella Issue989|
|2300|./algorithm|||||
|1879|./map|||||
|1541|./list|||||
|1190|./set|||||
|718|./deque|||||
|716|./vector|||||
|486|./cctype|$\textcolor{green}{\textsf{Done}}$|as part of [Issue1140](https://github.com/esbmc/esbmc/issues/1140)|N/A|[PR1141](https://github.com/esbmc/esbmc/pull/1141)|
|454|./locale|||||
|378|./ios|$\textcolor{green}{\textsf{Done}}$|Fixed MemberExpr assign|[Issue961](https://github.com/esbmc/esbmc/issues/961)(2x Issues)|[PR968](https://github.com/esbmc/esbmc/pull/968), [PR969](https://github.com/esbmc/esbmc/pull/969), [PR973](https://github.com/esbmc/esbmc/pull/973)|
|369|./iterator|||||
|346|./functional|||||
|324|./cstring|$\textcolor{green}{\textsf{Done}}$|Fixed parse errors|N/A|[PR960](https://github.com/esbmc/esbmc/pull/960)|
|281|./valarray|||||
|243|./cmath|$\textcolor{green}{\textsf{Done}}$|No fixes required|N/A|[PR993](https://github.com/esbmc/esbmc/pull/993)|
|242|./istream|$\textcolor{green}{\textsf{Done}}$|Fixed empty component name|[Issue975](https://github.com/esbmc/esbmc/issues/975)|[PR979](https://github.com/esbmc/esbmc/pull/979)|
|212|./tags|||||
|197|./fstream|||||
|167|./cstdlib|||||
|160|./sstream|||||
|151|./queue|||||
|135|./ostream|$\textcolor{green}{\textsf{Done}}$|No fixes required|N/A|[PR973](https://github.com/esbmc/esbmc/pull/973)|
|132|./memory|||||
|125|./stdexcept|$\textcolor{green}{\textsf{Done}}$|No fixes required|N/A|[PR958](https://github.com/esbmc/esbmc/pull/958)|
|123|./cstdio|$\textcolor{green}{\textsf{Done}}$|No fixes required|N/A|[PR944](https://github.com/esbmc/esbmc/pull/944)|
|115|./exception|$\textcolor{green}{\textsf{Done}}$|Fixed class ids for VFT and MemberExpr|medium fix, no issues raised|[PR947](https://github.com/esbmc/esbmc/pull/947), [PR948](https://github.com/esbmc/esbmc/pull/948)|
|110|./streambuf|$\textcolor{green}{\textsf{Done}}$|Fixed parse errors and pure virtual functions issues |medium fix, no issues raised|[PR954](https://github.com/esbmc/esbmc/pull/954)|
|106|./numeric|||||
|97|./complex|||||
|85|./stack|||||
|80|./bitset|||||
|67|./ctime|$\textcolor{green}{\textsf{Done}}$|as part of [Issue1140](https://github.com/esbmc/esbmc/issues/1140)|N/A|[PR1141](https://github.com/esbmc/esbmc/pull/1141)|
|64|./utility|$\textcolor{green}{\textsf{Done}}$|as part of [Issue1140](https://github.com/esbmc/esbmc/issues/1140)|N/A|[PR1141](https://github.com/esbmc/esbmc/pull/1141)|
|63|./limits|$\textcolor{green}{\textsf{Done}}$|as part of [Issue1140](https://github.com/esbmc/esbmc/issues/1140)|N/A|[PR1141](https://github.com/esbmc/esbmc/pull/1141)|
|59|./typeinfo|||||
|53|./new|$\textcolor{green}{\textsf{Done}}$|Fixes applied in OM, resolved conversion errors in frontend|[Issue987](https://github.com/esbmc/esbmc/issues/987)|[PR995](https://github.com/esbmc/esbmc/pull/995)|
|45|./csignal|$\textcolor{green}{\textsf{Done}}$|as part of [Issue1140](https://github.com/esbmc/esbmc/issues/1140)|N/A|[PR1141](https://github.com/esbmc/esbmc/pull/1141)|
|40|./systemc/communication/sc_signal_ports.h|Skipped|Not being used in our benchmarks|-|-|
|35|./systemc.h|Skipped|Not being used in our benchmarks|-|-|
|35|./ciso646|||||
|34|./float.h|$\textcolor{green}{\textsf{Done}}$|as part of [Issue1140](https://github.com/esbmc/esbmc/issues/1140)|N/A|[PR1141](https://github.com/esbmc/esbmc/pull/1141)|
|33|./systemc/kernel/sc_module.h|Skipped|Not being used in our benchmarks|-|-|
|31|./systemc/communication/sc_signal.h|Skipped|Not being used in our benchmarks|-|-|
|30|./iostream|$\textcolor{green}{\textsf{Done}}$|as part of [Issue1140](https://github.com/esbmc/esbmc/issues/1140)|N/A|[PR1141](https://github.com/esbmc/esbmc/pull/1141)|
|29|./cstdarg|$\textcolor{green}{\textsf{Done}}$|No fixes required|N/A|[PR943](https://github.com/esbmc/esbmc/pull/943)|
|28|./definitions.h|$\textcolor{green}{\textsf{Done}}$|Fixed parse errors|small fix, no issues raised|[PR958](https://github.com/esbmc/esbmc/pull/958)|
|27|./systemc/kernel/sc_sensitive.h|Skipped|Not being used in our benchmarks|-|-|
|27|./csetjmp|$\textcolor{green}{\textsf{Done}}$|as part of [Issue1140](https://github.com/esbmc/esbmc/issues/1140)|N/A|[PR1141](https://github.com/esbmc/esbmc/pull/1141)|
|23|./clocale|$\textcolor{green}{\textsf{Done}}$|as part of [Issue1140](https://github.com/esbmc/esbmc/issues/1140)|N/A|[PR1141](https://github.com/esbmc/esbmc/pull/1141)|
|20|./iomanip|$\textcolor{green}{\textsf{Done}}$|No fixes required|N/A|[PR973](https://github.com/esbmc/esbmc/pull/973)|
|20|./cassert|$\textcolor{green}{\textsf{Done}}$|No fixes required|N/A|[PR958](https://github.com/esbmc/esbmc/pull/958)|
|19|./pair|$\textcolor{green}{\textsf{Done}}$|as part of [Issue1140](https://github.com/esbmc/esbmc/issues/1140)|N/A|[PR1141](https://github.com/esbmc/esbmc/pull/1141)|
|16|./cfloat|$\textcolor{green}{\textsf{Done}}$|as part of [Issue1140](https://github.com/esbmc/esbmc/issues/1140)|N/A|[PR1141](https://github.com/esbmc/esbmc/pull/1141)|
|15|./climits|$\textcolor{green}{\textsf{Done}}$|as part of [Issue1140](https://github.com/esbmc/esbmc/issues/1140)|N/A|[PR1141](https://github.com/esbmc/esbmc/pull/1141)|
|15|./cerrno|$\textcolor{green}{\textsf{Done}}$|as part of [Issue1140](https://github.com/esbmc/esbmc/issues/1140)|N/A|[PR1141](https://github.com/esbmc/esbmc/pull/1141)|
|13|./libstl.cpp|Skipped|Not being used in our benchmarks|-|-|
|12|./systemc/communication/sc_clock.h|Skipped|Not being used in our benchmarks|-|-|
|9|./OM_compiler_defs.h|Skipped|Not an OM, just some compiler directives|||
|8|./systemc/kernel/sc_wait.h|Skipped|Not being used in our benchmarks|-|-|
|6|./unistd.h|$\textcolor{green}{\textsf{Done}}$|as part of [Issue1140](https://github.com/esbmc/esbmc/issues/1140), Only used in two TCs: try_catch/nec_ex10-io_[01/02]|N/A|[PR1141](https://github.com/esbmc/esbmc/pull/1141)|
|1|./string.h|$\textcolor{green}{\textsf{Done}}$|Just #include \<cstring\>|||
|1|./stdlib.h|||||
|1|./stdio.h|$\textcolor{green}{\textsf{Done}}$|as part of [Issue1140](https://github.com/esbmc/esbmc/issues/1140), Just #include "cstdio"|N/A|[PR1141](https://github.com/esbmc/esbmc/pull/1141)|
|1|./cstddef|$\textcolor{green}{\textsf{Done}}$|No fixes required|N/A|[PR943](https://github.com/esbmc/esbmc/pull/943)|


- 27/03 - 29/04: Completed 15/64 OMs, enabled 150 ish TCs, 3-4 OMs completed per week
- [Completed] Blocker ticket https://github.com/esbmc/esbmc/issues/989: ~~2-3 weeks to support Templates and resolve this umbrella ticker~~
  * Original Estimate: upto 3w, Actually Logged: 2w4d (see timestamps of the comments in issue 989)
- 11/07: Completed 41/64 OMs, 23 remain to fix. The remaining OMs are categorized below based on the size:   
  * Group 1 with LOC > 1000: 
    * string, algorithm, map, list, set​
  * Group 2 with 100 < LOC < 1000: 
    * deque, vector, cstdlib, locale, iterator, functional, valarray, tags, fstream, cstdlib, sstream, queue, memory, numeric
  * Group 3 with LOC < 100: 
    * complex, stack, bitset, typeinfo
- 05/04/2024: Completed 58/64 OMs, 6 remain to fix. The remaining OMs are categorized below based on the size:   
  * Group 1 with LOC > 1000: 
    * ~~string~~, ~~algorithm~~, ~~map~~, ~~list~~, ~~set~~​
  * Group 2 with 100 < LOC < 1000: 
    * ~~deque~~, ~~vector~~, ~~cstdlib~~, locale, ~~iterator~~, ~~functional~~, valarray, fstream, ~~cstdlib~~, sstream, ~~queue~~, ~~memory~~, ~~numeric~~
  * Group 3 with LOC < 100: 
    * ~~complex~~, ~~stack~~, bitset, ~~typeinfo~~
- 27/06/2024: Completed 63/64 OMs, 1 remain to fix. The remaining OMs are categorized below based on the size:   
  * Group 1 with LOC > 1000: 
    * ~~string~~, ~~algorithm~~, ~~map~~, ~~list~~, ~~set~~​
  * Group 2 with 100 < LOC < 1000: 
    * ~~deque~~, ~~vector~~, ~~cstdlib~~, locale, ~~iterator~~, ~~functional~~, ~~valarray~~, ~~fstream~~, ~~cstdlib~~, ~~sstream~~, ~~queue~~, ~~memory~~, ~~numeric~~
  * Group 3 with LOC < 100: 
    * ~~complex~~, ~~stack~~, ~~bitset~~, ~~typeinfo~~

## CUDA verification progress tracking: 
|LOC|OM file name|Status|Remarks|Issues Logged|PRs (fixes)|
|---|---|---|---|---|---|
|1850|./call_kernel.h|||||
|1167|./curand_kernel.h|exist_error||||
|1114|./cuda_runtime_api.h|||||
|829|./curand.h|exist error||||
|734|./sm_atomic_functions.h|No need to fix||||
|416|./vector_types.h|No need to fix||||
|314|./cuda.h|||||
|275|./driver_types.h|No need to fix||||
|174|./cuda_device_runtime_api.h|||||
|146|./device_launch_parameters.h|$\textcolor{orange}{\textsf{WIP}}$|function call: not enough arguments|||
|138|./host_defines.h|No need to fix||||
|9|./curand_precalc.h|No need to fix||||
|1|./builtin_types.h|No need to fix||||

CUDA OM issues:
- https://github.com/esbmc/esbmc/issues/1126
- https://github.com/esbmc/esbmc/issues/1176