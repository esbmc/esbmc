# SAFE-LD Implementation Plan: SMT-Based Formal Verification of Ladder Diagram Programs

**Status:** PLANNING  
**Project:** APP113435 — SAFE-LD (EPSRC Standard Research Grant)  
**Tracking:** umbrella issue TBD  
**Date:** 2026-06-09

---

## 1. Overview

SAFE-LD adds a new language front-end to ESBMC for IEC 61131-3 Ladder Diagram (LD) programs,
the most widely deployed PLC programming language. The front-end accepts vendor-neutral
**PLCopen XML** files exported from TIA Portal, Codesys, or Rockwell, translates them to
ANSI-C with `__ESBMC_assert()` annotations derived from a YAML property specification, and
feeds the resulting C file into the existing ESBMC verification pipeline unchanged.

The approach is semantics-driven: every LD construct is first given a formal meaning as a
**Structural Operational Semantics (SOS)** state-transition function over the PLC variable
store, and the C translation is derived systematically from that semantics. This grounds
translation correctness mathematically and distinguishes SAFE-LD from prior syntax-driven
approaches.

```
PLCopen XML  ──►  Parser  ──►  Semantic Analyser  ──►  IR
                                                        │
YAML props   ──────────────────────────────────────►  Property Encoder
                                                        │
                                    Code Generator ◄────┘
                                         │
                                    ANSI-C + __ESBMC_assert()
                                         │
                                    ESBMC (existing)
                                         │
                               ┌─────────┴──────────┐
                          Safety proof ✓       Counterexample ✗
                                              (JSON / ld-verify report)
```

The verification pipeline is exposed as `ld-verify`, a thin wrapper that orchestrates
the above steps and formats results for end users. SAFE-LD integrates as a new
`languaget` subclass and requires **no changes to the verification pipeline, solvers, or
symex**. Registering a new front-end does require small additions to ESBMC's language
dispatch layer (`src/langapi/mode.h`, `mode.cpp`, and `src/esbmc/globals.cpp`), exactly
as all other front-ends (Python, Jimple, Solidity) do — see §4.2.

---

## 2. Repository Structure

```
src/
└── ld-frontend/              # new directory — the entire SAFE-LD front-end
    ├── CMakeLists.txt
    ├── ld_language.h          # languaget subclass (mirrors python_language.h)
    ├── ld_language.cpp
    │
    ├── parser/                # WP2 / T2.1
    │   ├── plcopen_xml_parser.h
    │   ├── plcopen_xml_parser.cpp   # SAX/DOM over PLCopen XML schema
    │   ├── ld_ast.h                 # typed parse-tree node types
    │   └── ld_ast.cpp
    │
    ├── semantics/             # WP1 / T1.2 + WP2 / T2.1
    │   ├── sos_semantics.h          # SOS rule declarations
    │   ├── sos_semantics.cpp        # cyclic-scan state-transition functions
    │   ├── type_checker.h
    │   └── type_checker.cpp         # enforces IEC 61131-3 type rules
    │
    ├── ir/                    # WP2 / T2.2
    │   ├── ld_ir.h                  # IR node types (cyclic control-flow model)
    │   └── ld_ir.cpp
    │
    ├── codegen/               # WP2 / T2.2
    │   ├── c_codegen.h
    │   └── c_codegen.cpp            # IR → ANSI-C translation
    │
    ├── property/              # WP1 / T1.3 + WP2 / T2.2
    │   ├── yaml_property_parser.h
    │   ├── yaml_property_parser.cpp # YAML → property AST
    │   ├── property_encoder.h
    │   └── property_encoder.cpp     # property AST → __ESBMC_assert() calls
    │
    └── verify/                # WP2 / T2.3
        ├── ld_verify.h
        └── ld_verify.cpp            # ld-verify orchestration + JSON report

tools/
└── ld-verify/
    ├── CMakeLists.txt
    └── main.cpp               # ld-verify CLI entry point

docs/
├── safe-ld-implementation-plan.md   # this document
└── safe-ld-property-format.md       # YAML property spec (added in WP1)

regression/
└── ld/                        # all LD regression tests
    ├── CMakeLists.txt
    ├── benchmarks/            # WP3 benchmark suite (≥50 programs)
    │   ├── motor_interlock/   # CS1
    │   ├── conveyor_sequencing/ # CS2
    │   └── emergency_shutdown/  # CS3
    └── unit/                  # unit tests for each pipeline stage
        ├── parser/
        ├── semantics/
        ├── codegen/
        └── property/
```

---

## 3. Component Design

### 3.1 PLCopen XML Parser (`parser/`)

**Input:** PLCopen XML file (IEC 61131-10 exchange format).  
**Output:** Typed parse tree (`LdAst`) — a tree of `RungNode`, `ContactNode`, `CoilNode`,
`TimerFBNode`, `CounterFBNode`, `ArithFBNode`, and `NetworkNode` objects.

Key design points:

- Use a DOM parser (libxml2 — a **new** dependency, added via `find_package(LibXml2)`) to walk the XML tree.
- Normalize vendor-specific schema deviations (TIA Portal, Codesys, Rockwell) in a
  **schema normalisation layer** before constructing the AST. This directly mitigates the
  PLCopen XML schema-variation risk identified in the proposal.
- Every AST node carries a source location (`file`, `line`, `col`) to enable counterexample
  traces expressed in the original LD language.

### 3.2 Semantic Analyser (`semantics/`)

**Input:** `LdAst`  
**Output:** Validated `LdAst` annotated with resolved types, or a list of type errors.

The semantic analyser enforces:

1. **IEC 61131-3 type rules** — operand types match contact/coil polarity, timer preset
   types are `TIME`, counter presets are `INT`/`DINT`.
2. **Cyclic-scan execution model** — every rung's power-flow is evaluated in network order
   within each scan cycle; the analyser rejects programs whose data-flow would be
   order-dependent within a single scan (a common source of specification bugs).
3. **SOS consistency check** — the analyser validates the parse tree against the SOS
   specification produced in WP1. This makes the translation provably correct by
   construction: if the tree passes, the code generator can apply SOS rules mechanically.

### 3.3 Intermediate Representation (`ir/`)

**Input:** Validated `LdAst`  
**Output:** `LdIR` — a cyclic control-flow graph over SOS state-transition blocks.

The IR explicitly models the cyclic scan:

```
INIT_BLOCK          // PLC variable store initialisation
└── SCAN_LOOP       // while(true)
    ├── READ_INPUTS // capture physical inputs into I/O variables
    ├── RUNG_1      // evaluate rung 1: SOS step function
    ├── RUNG_2
    │   ⋮
    ├── RUNG_n
    └── WRITE_OUTPUTS // latch output coils to physical outputs
```

Each `RUNG_k` block is a sequence of `ContactEval`, `CoilAssign`, `FBCall` nodes that
directly correspond to SOS rule applications. This representation makes it straightforward
to prove the translation preserves the cyclic-scan semantics.

### 3.4 Code Generator (`codegen/`)

**Input:** `LdIR`  
**Output:** ANSI-C source file.

The translation follows the SOS state-transition functions mechanically:

| LD construct | Generated C |
|---|---|
| Normally-open contact `--[ ]--` | `bool pf = pf_in && var;` |
| Normally-closed contact `--[/]--` | `bool pf = pf_in && !var;` |
| Output coil `--( )--` | `var = pf;` |
| Set coil `--( S )--` | `if (pf) var = true;` |
| Reset coil `--( R )--` | `if (pf) var = false;` |
| TON timer | *(pseudocode sketch)* `if (pf) { if (!ton.IN) { ton.ET = 0; } ton.IN = true; ton.ET += SCAN_TIME; ton.Q = (ton.ET >= ton.PT); } else { ton.IN = false; ton.Q = false; ton.ET = 0; }` — full per-scan `ET` accumulation and `Q` logic are defined precisely in the SOS spec (T1.2) |
| CTU counter | `if (pf && !ctu_prev) ctu.CV++; if (ctu.CV >= ctu.PV) ctu.Q = true;` |

The scan loop is translated to a C `while(1)` loop. ESBMC's k-induction engine naturally
handles this loop structure; no special k-induction support is needed in the front-end.

A **fault injection mode** generates variants that introduce known semantic errors
(e.g., negated contact polarity) to validate that `ld-verify` detects them (WP1 validation
criteria).

### 3.5 Property Encoder (`property/`)

**Input:** YAML property specification file + validated `LdAst`.  
**Output:** `__ESBMC_assert()` call sites inserted into the generated C file.

The YAML format (specified in `docs/safe-ld-property-format.md`) supports the following
property classes, covering IEC 61508 safety requirements:

```yaml
properties:
  - id: P1
    kind: mutual_exclusion
    variables: [Motor_Forward, Motor_Reverse]
    description: "Forward and Reverse coils must never be energised simultaneously"

  - id: P2
    kind: response
    trigger: Start_Button
    response: Conveyor_Running
    max_scans: 2
    description: "Conveyor starts within 2 scan cycles of start button press"

  - id: P3
    kind: invariant
    expression: "ESD_Valve_Closed || !High_Pressure_Alarm"
    description: "ESD valve closes whenever high-pressure alarm is active"
```

Property kinds (WP1 taxonomy):

| Kind | IEC 61508 class | Generated assertion |
|---|---|---|
| `mutual_exclusion` | Safety integrity (independence) | `__ESBMC_assert(!(A && B), ...)` |
| `invariant` | Safety function activation | `__ESBMC_assert(expr, ...)` |
| `response` | Activation time | auxiliary scan-counter + assert |
| `absence` | Safe state persistence | `__ESBMC_assert(!expr, ...)` |
| `reachability` | Liveness | `__ESBMC_assert(false)` on target state |

The encoder inserts assertions at the end of every scan-loop iteration so ESBMC checks
them across all reachable scan sequences.

### 3.6 `ld-verify` Pipeline (`verify/` + `tools/ld-verify/`)

`ld-verify` is the end-to-end CLI tool:

```
ld-verify [options] <program.xml> [--props <props.yaml>]
```

Internally it:

1. Invokes the parser, semantic analyser, IR lowering, code generator, and property encoder
   to produce a self-contained `.c` file.
2. Calls `esbmc` on the `.c` file with the configured solver and strategy (default:
   `--k-induction --unlimited-k-steps --z3` with fallback to `--bmc --unwind 100`).
3. Parses ESBMC's output and emits a structured JSON report:

```json
{
  "result": "VIOLATION",
  "property": "P1",
  "description": "Forward and Reverse coils simultaneously energised",
  "counterexample": {
    "scan_cycle": 3,
    "rung": 7,
    "variable_store": { "Motor_Forward": true, "Motor_Reverse": true }
  }
}
```

Counterexample variable assignments are back-translated from C variable names to original
LD variable names using a symbol table built during code generation.

---

## 4. Integration with ESBMC Core

### 4.1 `languaget` Subclass

`ld_languaget` in `src/ld-frontend/ld_language.h` inherits from `languaget` (mirroring
`python_languaget`):

```cpp
class ld_languaget : public languaget
{
public:
  bool parse(const std::string &path) override;   // invoke PLCopen XML parser
  bool typecheck(contextt &, const std::string &) override; // semantic analysis
  bool final(contextt &) override;                // code generation + property encoding
  std::string id() const override { return "ld"; }
  void show_parse(std::ostream &) override;
  languaget *new_language() const override { return new ld_languaget; }
};
```

### 4.2 Language Dispatch Registration

ESBMC's dispatch (`language_id_by_path` in `src/langapi/mode.cpp`) is **extension-only**
— it matches on the file-name suffix and never inspects file contents. Registering on
`.xml` would therefore mis-route any XML file (SVCOMP witnesses, Jimple exports, etc.) to
the LD front-end. Instead:

- Register a dedicated **`.ld`** extension as the canonical input suffix.
- The `ld-verify` CLI can accept `.xml` files directly (bypassing `language_id_by_path`)
  and write a temporary `.ld`-suffixed copy before invoking ESBMC, or invoke
  `ld_languaget` directly without going through the extension-dispatch path.
- Users pass PLCopen XML files to `ld-verify`; only `ld-verify` (not bare `esbmc`)
  needs to handle `.xml` input.

The core changes required (mirroring the Python front-end addition):

1. `src/langapi/mode.h` — add `language_idt::LD` to the enum and declare
   `new_ld_language()`, `LANGAPI_MODE_LD`.
2. `src/langapi/mode.cpp` — add `extensions_ld[] = {"ld", nullptr}` and
   `language_desc_ld`.
3. `src/esbmc/globals.cpp` — add `LANGAPI_MODE_LD` inside
   `#ifdef ENABLE_LD_FRONTEND`.

### 4.3 CMake Integration

`src/ld-frontend/CMakeLists.txt`:

```cmake
find_package(LibXml2 REQUIRED)

add_library(ldfrontend STATIC
  ld_language.cpp
  parser/plcopen_xml_parser.cpp
  parser/ld_ast.cpp
  semantics/sos_semantics.cpp
  semantics/type_checker.cpp
  ir/ld_ir.cpp
  codegen/c_codegen.cpp
  property/yaml_property_parser.cpp
  property/property_encoder.cpp
  verify/ld_verify.cpp
)

target_include_directories(ldfrontend PUBLIC ${LIBXML2_INCLUDE_DIR})
target_link_libraries(ldfrontend PUBLIC ${LIBXML2_LIBRARIES} util)
```

`tools/ld-verify/CMakeLists.txt` links `ldfrontend` and produces the `ld-verify` binary.

### 4.4 Dependencies

| Dependency | Role | Already in ESBMC? |
|---|---|---|
| libxml2 | PLCopen XML DOM parsing | **No** — new dependency; add via `find_package(LibXml2 REQUIRED)` gated on `ENABLE_LD_FRONTEND` |
| yaml-cpp | YAML property file parsing | **Yes** — already required (`src/util/CMakeLists.txt` links `yaml-cpp::yaml-cpp`; `util/yaml_parser.h` exposes the interface) |
| nlohmann/json | JSON report output | **Yes** — already used by the Python frontend |

---

## 5. Work Packages and Implementation Tasks

### WP1 — Formal Semantics & Requirements (Months 1–6)

| Task | Output | Milestone |
|---|---|---|
| T1.1 Systematic Literature Review (PRISMA) | SLR report | — |
| T1.2 SOS specification of IEC 61131-3 LD | `docs/safe-ld-sos-semantics.md` + LaTeX formalisation | M1 (Month 3): SOS spec v1 complete |
| T1.3 Property taxonomy & YAML format | `docs/safe-ld-property-format.md`; 20 synthetic validation programs | M2 (Month 6): property format validated |

**M1 gate:** SOS spec covers contacts, coils, TON/TOF/TP timers, CTU/CTD counters,
arithmetic FBs, and the cyclic scan model; validated against IEC 61131-3 §2 by two
independent reviewers.

**M2 gate:** YAML format applied to 20 synthetic programs representing all property kinds;
all 20 programs pass semantic review; spec reviewed against IEC 61508 §7.

### WP2 — SAFE-LD Tool Development (Months 4–12)

| Task | Subtasks | Milestone |
|---|---|---|
| T2.1 Parser & Semantic Analyser | PLCopen XML parser; AST; type checker; SOS consistency check | M3 (Month 6): parser handles all WP1 SOS constructs |
| T2.2 Code Generator & Property Encoder | IR; C codegen; YAML parser; property encoder | M4 (Month 9): code generator correct on all benchmark programs |
| T2.3 ESBMC Integration & ld-verify | `ld_languaget`; CMake wiring; ld-verify CLI; JSON report | M5 (Month 12): end-to-end pipeline ready |
| T2.4 Test Suite (TDD, >90% coverage) | Unit tests per component; integration tests; fault-injection tests | tracked per task; coverage measured with gcov |

**Success criteria (WP2):**
- **Correctness:** ≥95% of benchmark programs translated with semantic equivalence verified
  by property checks and fault injection.
- **Performance:** average translation time <5 s for programs up to 1000 rungs.
- **Coverage:** >90% line coverage across `src/ld-frontend/`.

### WP3 — Industrial Validation (Months 10–24)

Three industrial case studies (CSs), each supplied as PLCopen XML programs by industry
collaborators or taken from published literature:

| CS | Program | Properties | Milestone |
|---|---|---|---|
| CS1 | Three-phase motor forward/reverse interlock | P1–P3: mutual exclusion of Forward/Reverse coils; interlock timing | M6 (Month 14): CS1 complete |
| CS2 | Multi-conveyor sequential startup with TON timer confirmation | P4–P6: startup sequencing; timer confirmation; belt speed safety | M7 (Month 18): CS2 complete |
| CS3 | Emergency Shutdown System (ESD) for process plant | P7–P10: immediacy, persistence, reset; SIL-2 properties | M8 (Month 22): CS3 + comparative analysis complete |

**Comparative benchmarking (T3.4):** All ≥50 benchmark programs run against nuXmv (the
primary comparator). Metrics: verification coverage, analysis time, false-positive /
false-negative rates, counterexample quality. nuXmv was selected because it is the
maintained successor to NuSMV, which was used in the closest prior industrial LD
verification study.

**Benchmark dataset (T3.5):** Released as `regression/ld/benchmarks/` with each program,
its YAML property file, and the expected `ld-verify` verdict. Submitted as SV-COMP
category proposal (T4.5).

### WP4 — Dissemination & Extension (Months 25–36)

| Task | Output | Milestone |
|---|---|---|
| T4.1 Paper 1: Semantics + Tool | Journal article (IEEE Transactions on Industrial Informatics target) | M9 (Month 28): submitted |
| T4.2 Paper 2: CSs + Comparison | Journal article (TACAS / CAV / ISSTA target) | M10 (Month 32): submitted |
| T4.3 LLM Property Generation (exploratory) | Prototype + preliminary empirical result; not a production feature | — |
| T4.4 Open-Source Release | SAFE-LD + ld-verify tagged release; TIA Portal + Codesys integration guides | M11 (Month 36): full open-source release |
| T4.5 SV-COMP Category Proposal | Submission to SV-COMP steering committee | M11 (Month 36) |

---

## 6. Testing Strategy

### Unit Tests

Each pipeline stage has a dedicated unit-test suite under `regression/ld/unit/`:

- **Parser:** round-trip tests (parse → serialise → compare); malformed XML rejection;
  schema normalisation for each vendor export format.
- **Semantics:** type-error detection on crafted invalid programs; SOS consistency
  acceptance on all WP1 synthetic programs.
- **Code generator:** output C file compiles clean with `-Wall -Wextra`; fault-injection
  variants detected by `ld-verify`.
- **Property encoder:** each property kind generates the correct `__ESBMC_assert()` call;
  vacuous assertions (always true/false) flagged.

### Integration Tests

Full `ld-verify` end-to-end tests for every benchmark program, with expected verdict
checked by CTest. Added to the ESBMC CI matrix alongside the existing regression suites.

### Fault Injection Validation (WP1 gate)

For each SOS rule, a known semantic error is introduced into a synthetic program and
`ld-verify` must produce a `VIOLATION` result naming the correct property. This validates
both the translation and the verifier on real semantic errors, not just syntactic ones.

---

## 7. Risk Mitigations

| Risk | Mitigation (from proposal) | Implementation note |
|---|---|---|
| PLCopen XML schema variation | Schema normalisation layer in `parser/` | Tested against TIA Portal and Codesys exports in WP1 |
| k-induction non-termination on timer-heavy programs | Configurable `--unwind` bound; portfolio solver fallback | Exposed as `ld-verify --strategy bmc\|k-induction\|portfolio` |
| PDRA recruitment delay | Co-I bridges short-term | No implementation impact; timeline padded by 1 month per WP |
| Industrial programs not in PLCopen XML | Synthetic programs from published CSs; team has Codesys + TIA Portal access | WP3 CS programs collected in Month 10 |
| Scope underestimation for full LD coverage | WP1 property taxonomy explicitly bounds scope; graceful unsupported-construct errors | Parser emits a structured `UnsupportedConstruct` error rather than silently mishandling |

---

## 8. Key Design Decisions

1. **Minimal ESBMC core changes; no pipeline modifications.** SAFE-LD produces standard
   ANSI-C with `__ESBMC_assert()` and hands off to the existing verification pipeline
   unchanged. Registering the new front-end requires the same small additions to
   `mode.h`, `mode.cpp`, and `globals.cpp` that every other ESBMC front-end requires
   (Python, Jimple, Solidity — see §4.2). The verification pipeline, solvers, symex,
   and GOTO-program IR are not touched. This eliminates integration risk and immediately
   delivers k-induction, multi-solver portfolio, and witness generation without any
   front-end work.

2. **Semantics-driven translation.** The SOS specification is the primary design artefact.
   The parser, IR, and code generator are all derived from it. This provides a mathematical
   correctness argument that syntax-driven translators cannot offer, and it structures WP1
   (semantics) as a prerequisite for WP2 (tool) rather than an afterthought.

3. **Vendor-neutral input via PLCopen XML.** No vendor SDK is required; programs are
   exported as PLCopen XML from any IEC 61131-3 IDE. The schema normalisation layer absorbs
   vendor differences once, keeping the rest of the pipeline vendor-agnostic.

4. **YAML property specification.** Safety engineers express properties in domain vocabulary
   (variable names, scan counts) rather than temporal logic. The property encoder handles
   the mapping to `__ESBMC_assert()` automatically, lowering the expertise barrier for
   industrial adoption.

5. **Back-translated counterexamples.** The symbol table built during code generation maps
   every C variable back to its LD name and rung. ESBMC's counterexample trace is therefore
   presented in LD terms (scan cycle, rung, variable store), not in C terms, which is
   essential for practitioner usability.

---

## 9. Milestones Summary

| ID | Month | Description |
|---|---|---|
| M1 | 3 | SOS specification v1 complete |
| M2 | 6 | Property format validated against 20 synthetic programs |
| M3 | 6 | Parser handles all WP1 SOS constructs |
| M4 | 9 | Code generator correct on all benchmark programs |
| M5 | 12 | End-to-end `ld-verify` pipeline ready |
| M6 | 14 | CS1 (motor interlock) complete |
| M7 | 18 | CS2 (conveyor sequencing) complete |
| M8 | 22 | CS3 (ESD) + comparative analysis complete |
| M9 | 28 | Paper 1 submitted |
| M10 | 32 | Paper 2 submitted |
| M11 | 36 | Full open-source release + SV-COMP category proposal |
