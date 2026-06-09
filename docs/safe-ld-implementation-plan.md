# SAFE-LD Implementation Plan: SMT-Based Formal Verification of Ladder Diagram Programs

**Status:** PLANNING  
**Project:** APP113435 — SAFE-LD (EPSRC Standard Research Grant)  
**Tracking:** umbrella issue TBD  
**Date:** 2026-06-09

---

## 1. Overview

SAFE-LD adds a new language front-end to ESBMC for IEC 61131-3 Ladder Diagram (LD) programs,
the most widely deployed PLC programming language. The front-end accepts vendor-neutral
**PLCopen XML** files exported from TIA Portal, Codesys, or Rockwell and translates them
**directly into ESBMC's GOTO IR** (via the irep2 type system), with safety assertions
derived from a YAML property specification encoded as native `code_assertt` nodes. No
intermediate C file is produced.

The approach is semantics-driven: every LD construct is first given a formal meaning as a
**Structural Operational Semantics (SOS)** state-transition function over the PLC variable
store, and the GOTO IR is derived systematically from that semantics. This grounds
translation correctness mathematically, reduces reliance on unverified translation
components, and distinguishes SAFE-LD from prior syntax-driven approaches.

```
PLCopen XML  ──►  Parser  ──►  Semantic Analyser  ──►  LdIR
                                                         │
YAML props   ────────────────────────────────────►  Property Encoder
                                                         │
                                     GOTO IR Generator ◄─┘
                                              │
                               GOTO IR (irep2 symbolt / code_blockt)
                                              │
                                    ESBMC verification engine
                                              │
                               ┌──────────────┴──────────────┐
                          Safety proof ✓              Counterexample ✗
                                                  (LD-native JSON report)
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
    │   ├── ld_ir.h                  # LdIR node types (cyclic control-flow model)
    │   └── ld_ir.cpp
    │
    ├── ir_gen/                # WP2 / T2.2
    │   ├── ld_converter.h
    │   └── ld_converter.cpp         # LdIR → GOTO IR (irep2 symbolt / code_blockt)
    │
    ├── property/              # WP1 / T1.3 + WP2 / T2.2
    │   ├── yaml_property_parser.h
    │   ├── yaml_property_parser.cpp # YAML → property AST
    │   ├── property_encoder.h
    │   └── property_encoder.cpp     # property AST → code_assertt nodes in GOTO IR
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
        ├── ir_gen/
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

### 3.4 GOTO IR Generator (`ir_gen/`)

**Input:** `LdIR` + property `code_assertt` nodes from the property encoder.  
**Output:** Populated `contextt` (ESBMC symbol table + GOTO function bodies).

`ld_converter` follows the same pattern as `python_converter`: it builds `symbolt` entries
and `codet` trees directly using ESBMC's irep2 types, then inserts them into the `contextt`
passed in by `ld_languaget::typecheck()`. No C file is produced at any stage.

**Symbol table construction.** Every LD variable (BOOL contact/coil, TIME timer field,
INT/DINT counter field) becomes a `symbolt` with:
- `type` drawn from `bool_type()`, `uint_type(32)`, etc.
- `location` set to the originating PLCopen XML file/line/col.
- `base_name` equal to the PLCopen XML variable identifier; `name` prefixed with
  `ld::` to avoid clashes.

**Scan-loop function.** The converter emits a single `__ESBMC_main`-equivalent function
whose body is a `code_whilet(true_exprt(), scan_body)`, where `scan_body` is a
`code_blockt` containing one `code_blockt` per rung. This directly models the IEC 61131-3
cyclic scan without requiring ESBMC to see a C `while(1)`.

**Per-rung translation.** Each `LdIR` rung block maps mechanically to irep2 nodes
following the SOS state-transition rules:

| LD construct | GOTO IR node |
|---|---|
| Normally-open contact `--[ ]--` | `and_exprt(pf_in, symbol_exprt(var))` |
| Normally-closed contact `--[/]--` | `and_exprt(pf_in, not_exprt(symbol_exprt(var)))` |
| Output coil `--( )--` | `code_assignt(symbol_exprt(var), pf)` |
| Set coil `--( S )--` | `code_ifthenelset(pf, code_assignt(var, true_exprt()))` |
| Reset coil `--( R )--` | `code_ifthenelset(pf, code_assignt(var, false_exprt()))` |
| TON timer *(per-scan step)* | sequence of `code_ifthenelset` + `code_assignt` for `IN`, `ET`, `Q` fields — full logic defined in SOS spec (T1.2) |
| CTU counter *(per-scan step)* | `code_ifthenelset` on rising edge → increment `CV`; `code_assignt` of `Q = (CV >= PV)` |

**Fault injection mode.** An optional converter flag negates selected contact polarities
or skips coil assignments to produce known-faulty GOTO programs. Used in WP1 validation
to confirm `ld-verify` detects each planted semantic error.

### 3.5 Property Encoder (`property/`)

**Input:** YAML property specification file + validated `LdAst`.  
**Output:** `code_assertt` nodes to be appended to the scan-loop body by the GOTO IR
generator (§3.4).

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

| Kind | IEC 61508 class | GOTO IR node emitted |
|---|---|---|
| `mutual_exclusion` | Safety integrity (independence) | `code_assertt(not_exprt(and_exprt(A, B)))` |
| `invariant` | Safety function activation | `code_assertt(expr)` |
| `response` | Activation time | auxiliary scan-counter `symbolt` + `code_assertt` on counter bound |
| `absence` | Safe state persistence | `code_assertt(not_exprt(expr))` |
| `reachability` | Liveness | `code_assertt(false_exprt())` on target-state guard |

Each `code_assertt` node carries:
- `location` referencing the YAML property file and property id.
- `comment` set to the property `description` field so ESBMC's counterexample output
  names the violated property in human-readable form.

The encoder appends all `code_assertt` nodes at the end of every scan-loop iteration
body so ESBMC checks them across all reachable scan sequences.

**Soundness and completeness.** Property kinds differ in their verification guarantees:

| Kind | Sound? | Complete? | Condition |
|---|---|---|---|
| `mutual_exclusion` | Yes | Yes | Checked at every scan; k-induction or BMC both give exact results |
| `invariant` | Yes | Yes | Same as above |
| `absence` | Yes | Yes | Same as above |
| `response` | Yes | **Bounded** | Sound only if `max_scans` is a valid upper bound on required response time. If the system can legitimately respond in > `max_scans` cycles the encoding is a false alarm. The YAML value must be justified by timing analysis or IEC 61508 §7 requirements. |
| `reachability` | Yes (k-ind) / Bounded (BMC) | No | Under k-induction, proving a state unreachable is sound and complete for the cyclic-scan model. Under BMC, only unreachability up to the unwind bound is established. |

Properties with "Yes / Yes" guarantees should be preferred for safety-critical properties.
`response` and `reachability` properties must be annotated in the YAML file with a
`justification` field recording the bound rationale; `ld-verify` will reject them without
one.

### 3.6 `ld-verify` Pipeline (`verify/` + `tools/ld-verify/`)

`ld-verify` is the end-to-end CLI tool:

```
ld-verify [options] <program.xml> [--props <props.yaml>]
```

Internally it invokes `esbmc` with the `.ld`-renamed input file and the configured
strategy (default: `--k-induction --unlimited-k-steps --z3` with fallback to
`--bmc --unwind 100`). Because SAFE-LD generates GOTO IR directly, ESBMC's clang
front-end is **never invoked** — `ld_languaget::typecheck()` populates the `contextt`
and control passes straight to symex.

`ld-verify` then parses ESBMC's output and emits a structured JSON report:

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

Because every `symbolt` and `code_assertt` node was created with LD source locations
and LD variable names, the counterexample trace produced by ESBMC already references
the original rung numbers and PLCopen XML identifiers. No back-translation table is
needed.

### 3.7 Translation Correctness

This section defines the formal guarantee that `ld_converter` is expected to satisfy
and outlines the proof strategy. The guarantee is stated as a semantic preservation
theorem; it is the obligation that makes SAFE-LD a formal tool rather than a
best-effort translator.

#### Semantic Preservation Theorem

Let P be a valid PLCopen XML program and σ₀ ∈ Σ an initial PLC variable store.
Let ⟨P, σ⟩ →_SOS σ' denote one full scan-cycle step under the SOS state-transition
rules (T1.2). Let G(P) be the GOTO program produced by `ld_converter(P)`, and let
s₀ ∈ S be the corresponding initial GOTO state.

**Theorem.** For every n ≥ 0, the variable-store snapshot at the start of scan cycle n
in the SOS trace equals the projection of the GOTO state at the start of the n-th
scan-loop iteration onto the LD variables.

More precisely, define the relation R ⊆ Σ × S by:

> (σ, s) ∈ R iff for every LD variable v, σ(v) = s(`ld::v`)

Then:

1. **(Initialisation)** (σ₀, s₀) ∈ R.
2. **(Step preservation)** If (σ, s) ∈ R and ⟨P, σ⟩ →_SOS σ', and s' is the GOTO
   state after one complete execution of the scan-loop body from s, then (σ', s') ∈ R.

#### Proof Strategy

Step preservation is proved by **structural induction on rung order**, with each rung
proved by **case analysis on the LdIR node type**. For each case the proof obligation
is: given (σ, s) ∈ R and the rung's SOS rule, show the GOTO IR instructions generated
by `ld_converter` for that node produce s' such that (σ', s') ∈ R.

For contacts and coils the obligation is discharged by direct inspection of the
`and_exprt` / `not_exprt` / `code_assignt` node generated (§3.4 table).

For FB constructs (TON, CTU) the obligation is non-trivial: it requires showing the
multi-instruction GOTO encoding — `code_ifthenelset` chains over `IN`, `ET`, `Q` — matches
the SOS state-machine step function defined in T1.2. This is the primary proof
obligation of WP2. Validation by fault injection (§3.4 and §6) provides executable
evidence prior to a formal proof.

#### What is Formally Guaranteed

- Any `VIOLATION` result from ESBMC corresponds to a genuine violation of the
  SOS-level assertion: a scan sequence exists in which the SOS semantics violate the
  specified safety property.
- A `VERIFICATION SUCCESSFUL` result from k-induction is a proof that no such sequence
  exists (up to the correctness of `ld_converter` and ESBMC's symex).

#### What is Not Guaranteed

- **Completeness of BMC mode.** Bounded model checking checks up to a finite unwind
  depth. A violation requiring more scan cycles than the bound will be missed; the
  result should be reported as `INCOMPLETE`, not `SAFE`.
- **Soundness of bounded `response` properties.** See §3.5 for the bound-justification
  requirement.
- **Correctness of the SOS specification.** The SOS spec (T1.2) is validated by review
  and fault injection but is not itself formally proven against the IEC 61131-3 normative
  text. It is the assumed semantic ground truth for the theorem above.

---

## 4. Integration with ESBMC Core

### 4.1 `languaget` Subclass

`ld_languaget` in `src/ld-frontend/ld_language.h` inherits from `languaget` (mirroring
`python_languaget`):

```cpp
class ld_languaget : public languaget
{
public:
  // Parse PLCopen XML → LdAst (stored in member); run semantic analyser.
  bool parse(const std::string &path) override;

  // Run ld_converter: populate contextt with symbolt entries and the
  // scan-loop GOTO function body. This is where all IR generation happens,
  // mirroring python_languaget::typecheck() calling python_converter::convert().
  bool typecheck(contextt &context, const std::string &module) override;

  bool final(contextt &) override { return false; }
  std::string id() const override { return "ld"; }
  void show_parse(std::ostream &) override;
  languaget *new_language() const override { return new ld_languaget; }

private:
  LdAst ast_;
  std::string props_path_; // set from --ld-props CLI option
};
```

The division of responsibilities mirrors the Python frontend: `parse()` produces the
validated AST; `typecheck()` drives `ld_converter`, which fills the `contextt` with all
symbols and the main scan-loop function; `final()` is a no-op.

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
  ir_gen/ld_converter.cpp
  property/yaml_property_parser.cpp
  property/property_encoder.cpp
  verify/ld_verify.cpp
)

target_include_directories(ldfrontend PUBLIC ${LIBXML2_INCLUDE_DIR})
# irep2 and util are already linked transitively via the ESBMC build graph;
# explicit linkage follows the python-frontend pattern.
target_link_libraries(ldfrontend PUBLIC ${LIBXML2_LIBRARIES} util irep2)
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
| T2.2 GOTO IR Generator & Property Encoder | LdIR; `ld_converter` (irep2); YAML parser; property encoder (`code_assertt`) | M4 (Month 9): IR generator correct on all benchmark programs |
| T2.3 ESBMC Integration & ld-verify | `ld_languaget`; CMake wiring; ld-verify CLI; JSON report | M5 (Month 12): end-to-end pipeline ready |
| T2.4 Test Suite (TDD, >90% coverage) | Unit tests per component; integration tests; fault-injection tests | tracked per task; coverage measured with gcov |

**Success criteria (WP2):**
- **Correctness:** ≥95% of benchmark programs translated to GOTO IR with semantic
  equivalence verified by property checks and fault injection.
- **Performance:** average end-to-end `ld-verify` time <5 s for programs up to
  1000 rungs. Justified by the structural properties of PLC programs: the scan body
  is finite-state per iteration (no heap allocation, no recursion); industrial programs
  typically have <500 boolean variables; and the cyclic-scan loop structure means
  k-induction convergence is governed by the depth of control-flow nesting within a
  single rung, not by the number of rungs. The main exception is timer-heavy programs
  (see §7 risk mitigations).
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
- **GOTO IR generator:** each `LdIR` node maps to the expected irep2 `codet`/`exprt`
  type; the emitted `contextt` passes ESBMC's `clang_cpp_adjust` equivalent without
  errors; fault-injection variants produce a `VIOLATION` verdict.
- **Property encoder:** each property kind produces a `code_assertt` with the correct
  guard expression and location; vacuous assertions (always true/false) are flagged.

### Integration Tests

Full `ld-verify` end-to-end tests for every benchmark program, with expected verdict
checked by CTest. Added to the ESBMC CI matrix alongside the existing regression suites.

### Fault Injection Validation (WP1 gate)

For each SOS rule, a known semantic error is introduced into a synthetic program and
`ld-verify` must produce a `VIOLATION` result naming the correct property. This validates
both the translation and the verifier on real semantic errors, not just syntactic ones.

---

## 7. Risk Mitigations

| Risk | Mitigation | Implementation note |
|---|---|---|
| PLCopen XML schema variation between vendors | Schema normalisation layer in `parser/` | Tested against TIA Portal, Codesys, and Rockwell exports in WP1; vendor-specific test programs kept in `regression/ld/` |
| k-induction non-termination on timer-heavy programs | TON/TOF/TP timer state abstraction: `Q` modelled as nondet bool constrained by `__ESBMC_assume` to SOS timer invariants, reducing required induction depth to O(1); full concrete encoding retained as an option | Fallback exposed as `ld-verify --strategy bmc\|k-induction\|portfolio\|abstract-timers`; portfolio mode applies per-program timeout (default 60 s) and reports `INCOMPLETE` rather than hanging |
| Solver timeout cascade in benchmark runs | Per-program timeout in `ld-verify` (default: 60 s); aggregate benchmark runner collects partial results and reports coverage fraction | `TIMEOUT` verdict treated as `UNKNOWN` in benchmark statistics; not counted as false positive or false negative |
| Unsupported LD constructs accumulation | Tiered support plan: **Tier 1** (WP2 scope) — contacts, coils, TON/TOF/TP, CTU/CTD, arithmetic FBs; **Tier 2** (post-project) — advanced FBs, structured text inline, arrays; **Tier 3** — vendor-specific extensions. Each unsupported construct emits a structured `UnsupportedConstruct(name, tier)` error, not a silent failure. | WP1 property taxonomy explicitly fixes the Tier 1 boundary; any Tier 2+ construct encountered in WP3 case studies is recorded as a known limitation in the paper |
| Incomplete PLCopen XML exports (missing FB declarations, partial networks) | Strict schema validation at parse time with diagnostic messages naming the missing element and the expected schema location | A library of known-valid exports from each vendor is maintained in `regression/ld/`; WP3 programs validated against the library before industrial use |
| Semantic drift across vendors (differing interpretations of IEC 61131-3 edge cases) | Vendor-specific SOS annotations in T1.2 document known divergences; regression tests cover each documented divergence | Divergences that affect verification results are flagged in `ld-verify` output with a `vendor-note` field |
| PDRA recruitment delay | Co-I bridges short-term | No implementation impact; timeline padded by 1 month per WP |
| Industrial programs not in PLCopen XML | Synthetic programs from published CSs; team has Codesys and TIA Portal access | WP3 CS programs collected in Month 10 |

---

## 8. Key Design Decisions

1. **Direct GOTO IR generation; no C intermediary.** SAFE-LD's `ld_converter` populates
   ESBMC's `contextt` directly with `symbolt` entries and `codet` trees, following the
   same pattern as `python_converter`. ESBMC's clang front-end is never invoked. This
   significantly reduces reliance on unverified translation components: the path from LD
   semantics to the verifier is SOS specification → `ld_converter` → symex, with no C
   compilation step in between (the trusted base still includes `ld_converter` itself,
   ESBMC's symex, and the SMT solvers). Registering the front-end requires the same small
   additions to `mode.h`, `mode.cpp`, and `globals.cpp` that every other ESBMC front-end
   requires (Python, Jimple, Solidity — see §4.2). The verification pipeline, solvers,
   and symex are not touched.

2. **Semantics-driven translation.** The SOS specification is the primary design artefact.
   The parser, IR, and code generator are all derived from it. This provides a mathematical
   correctness argument that syntax-driven translators cannot offer, and it structures WP1
   (semantics) as a prerequisite for WP2 (tool) rather than an afterthought.

3. **Vendor-neutral input via PLCopen XML.** No vendor SDK is required; programs are
   exported as PLCopen XML from any IEC 61131-3 IDE. The schema normalisation layer absorbs
   vendor differences once, keeping the rest of the pipeline vendor-agnostic.

4. **YAML property specification.** Safety engineers express properties in domain vocabulary
   (variable names, scan counts) rather than temporal logic. The property encoder handles
   the mapping to `code_assertt` nodes automatically, lowering the expertise barrier for
   industrial adoption.

5. **Native LD counterexamples.** Because every `symbolt` is created with its PLCopen XML
   identifier as `base_name` and every `code_assertt` carries the originating LD source
   location, ESBMC's counterexample trace already references rung numbers and variable
   names from the LD program directly. No back-translation step is needed, and the
   structured JSON report is produced by reading ESBMC's native output rather than
   remapping from C names.

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
