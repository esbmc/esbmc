# ESBMC-PLC Implementation Plan: SMT-Based Formal Verification of Ladder Diagram Programs

**Status:** PLANNING — WP2 largely implemented (skeleton #5280, property pipeline #5289, ld-verify runner + fault injection #5294, industrial benchmarks #5427, user-FB/REAL/watchdog #5620, graphical-LD soundness fixes + WP1 SOS spec)  
**Project:** APP113435 — ESBMC-PLC (EPSRC Standard Research Grant)  
**Tracking:** umbrella issue TBD  
**Date:** 2026-06-09 (status refreshed 2026-07-30)

> **Implementation note.** The boolean/combinational Tier-1 subset, the
> integer-arithmetic constructs (TON/TOF/TP timers, CTU/CTD counters, and the
> `response` property), and — beyond the original Tier-1 scope — user-defined
> function-block bodies, REAL/analog process variables, and an optional
> scan-watchdog all now lower to GOTO IR and verify end-to-end (see §10). The
> suite under `regression/ld/` has grown to 37 CTest cases, plus 10 for the
> `ld-verify` runner, and CI now actually runs them. All four curated
> benchmarks have validated verdicts and are wired as regression tests. The WP1
> SOS specification exists as `docs/safe-ld-sos-semantics.md`; its independent
> M1 review is still outstanding.

---

## 1. Overview

ESBMC-PLC adds a new language front-end to ESBMC for IEC 61131-3 Ladder Diagram (LD) programs,
the most widely deployed PLC programming language. The front-end accepts vendor-neutral
**PLCopen XML** files exported from TIA Portal, Codesys, or Rockwell and translates them
**directly into ESBMC's GOTO IR** (via the irep2 type system), with safety assertions
derived from a YAML property specification encoded as native `code_assertt` nodes. No
intermediate C file is produced.

The approach is semantics-driven: every LD construct is first given a formal meaning as a
**Structural Operational Semantics (SOS)** state-transition function over the PLC variable
store, and the GOTO IR is derived systematically from that semantics. This grounds
translation correctness mathematically, reduces reliance on unverified translation
components, and distinguishes ESBMC-PLC from prior syntax-driven approaches.

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
the above steps and formats results for end users. ESBMC-PLC integrates as a new
`languaget` subclass and requires **no changes to the verification pipeline, solvers, or
symex**. Registering a new front-end does require small additions to ESBMC's language
dispatch layer (`src/langapi/mode.h`, `mode.cpp`, and `src/esbmc/globals.cpp`), exactly
as all other front-ends (Python, Jimple, Solidity) do — see §4.2.

---

## 2. Repository Structure

```
src/
└── ld-frontend/              # new directory — the entire ESBMC-PLC front-end
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
    │   ├── ld_ir.cpp
    │   ├── ld_ir_builder.h          # LdAst → LdIR lowering
    │   └── ld_ir_builder.cpp
    │
    ├── ir_gen/                # WP2 / T2.2
    │   ├── ld_converter.h
    │   ├── ld_converter.cpp         # LdIR → GOTO IR (irep2 symbolt / code_blockt)
    │   ├── st_fb_translator.h       # Structured Text FB bodies → codet
    │   └── st_fb_translator.cpp
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
├── roadmap/
│   └── safe-ld-implementation-plan.md   # this document
├── safe-ld-property-format.md           # YAML property spec (WP1 / T1.3)
└── safe-ld-sos-semantics.md             # SOS specification (WP1 / T1.2)

regression/
└── ld/                        # driver-level LD regression tests (one dir each)
    ├── CMakeLists.txt
    └── benchmarks/            # WP3 benchmark dataset, excluded from CTest
        ├── motor_interlock/     # CS1
        ├── conveyor_sequencing/ # CS2
        ├── emergency_shutdown/  # CS3
        ├── stairs_light/        # CONTROLLINO / OpenPLC, graphical
        └── water_control/       # CONTROLLINO / OpenPLC, graphical

unit/
└── ld-frontend/               # Catch2 unit tests per pipeline stage
    ├── test_parser.cpp
    ├── test_property.cpp
    ├── test_ir_gen.cpp
    └── fixtures/
```

---

## 3. Component Design

### 3.1 PLCopen XML Parser (`parser/`)

**Input:** PLCopen XML file (IEC 61131-10 exchange format).  
**Output:** Typed parse tree (`LdAst`) — a tree of `RungNode`, `ContactNode`, `CoilNode`,
`TimerFBNode`, `CounterFBNode`, `ArithFBNode`, and `NetworkNode` objects.

Key design points:

- Use a DOM parser (**pugixml** — a **new** dependency, bundled via CMake `FetchContent`) to walk the XML tree. pugixml is a single-header + single-`.cpp` library with no system dependencies, linking cleanly as a static target on macOS, Windows, and Linux without pkg-config or ABI concerns.
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

**Execution model scope (Tier 1).** The IR models a **strictly synchronous, single-task**
cyclic scan: one periodic task, no I/O interrupt tasks, no multi-task PLC configurations,
no pre-emptive scheduling. This is the correct scope for the IEC 61131-3 safety properties
targeted by ESBMC-PLC. Programs containing interrupt task declarations or multi-task
configurations are rejected by the semantic analyser with a structured
`UnsupportedConstruct(InterruptTask, tier=2)` error. Multi-task support is Tier 2; if
addressed in future work, ESBMC's existing concurrency primitives would be the integration
point. The single `code_whilet(true_exprt(), scan_body)` model must not be used to
represent concurrent tasks.

**Synchronous fixed-tick time model.** Rather than tracking wall-clock time (which causes
state-space explosion in SMT), the IR uses an **abstract tick model**: every scan-loop
iteration advances time by exactly **one tick**. Timer preset values (`PT`) are
dimensionless tick counts. This gives a fully concrete, deterministic time encoding:

- `ET` is incremented by 1 per scan while `IN = true` — encoded as `plus_exprt(ET, one)`.
- `Q` is assigned `geq_exprt(ET, PT)` at the end of each scan.

No nondeterminism in time progression and no `__ESBMC_assume` needed. This eliminates
the vacuous-failure risk that arises when Δt is unconstrained: a TON timer with `PT = N`
fires after exactly N scan iterations, making `response` properties and timer-dependent
invariants checkable with a known induction depth of N. Wall-clock jitter is out of scope
(see §3.7 "What is Not Guaranteed").

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
| Rising-edge contact `--[P]--` | `and_exprt(pf_in, and_exprt(var, not_exprt(prev)))`, with `prev` latched in the scan epilogue |
| TON timer *(fixed-tick model)* | `code_ifthenelset(IN, ET := ET+1, ET := 0)`; `code_assignt(Q, and_exprt(IN, geq_exprt(ET, PT)))` — fires deterministically after exactly `PT` scan ticks |
| TOF timer *(fixed-tick model)* | `code_ifthenelset(IN, {ET := 0; Q := true}, code_ifthenelset(Q, {ET := ET+1; Q := ET < PT}))` — holds `Q` for `PT` ticks after `IN` drops, and stays off until first enabled |
| CTU counter *(per-scan step)* | `code_ifthenelset` on rising edge → increment `CV`; `code_assignt` of `Q = (CV >= PV)` |

The full step functions, and the rules for graphical-network resolution and
feedback variables, are specified in `docs/safe-ld-sos-semantics.md` (T1.2).

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
strategy. Because ESBMC-PLC generates GOTO IR directly, ESBMC's clang front-end is
**never invoked** — `ld_languaget::typecheck()` populates the `contextt` and
control passes straight to symex.

**As implemented (#5289).** `LdVerifyRunner::run()`:

- Stages a non-`.ld` input (e.g. PLCopen `.xml`) into a PID-tagged temporary
  `.ld` copy so ESBMC's extension dispatch (§4.2) routes it to the LD front-end,
  and removes the copy afterwards.
- Locates the `esbmc` binary via `$ESBMC`, else `$PATH`.
- Maps `--strategy`: `k-induction` → `--k-induction --unlimited-k-steps`;
  `bmc` → `--incremental-bmc --unwind <N>`. Only those two are accepted;
  anything else — including `portfolio` and `abstract-timers`, which §7 lists
  as intended fallbacks but which are not implemented — is rejected as
  `ERROR`.
- Passes the YAML file through as `--ld-props <file>`.
- Parses the verdict and emits the JSON report, with the verdict set
  `{SAFE, VIOLATION, INCOMPLETE, UNKNOWN, ERROR}`. `VERIFICATION SUCCESSFUL`
  maps to `SAFE` and `VERIFICATION FAILED` to `VIOLATION`; `VERIFICATION
  UNKNOWN` maps to `INCOMPLETE` under `bmc` and to `UNKNOWN` under
  `k-induction`, where it means the proof did not converge. Output with no
  verdict line at all is reported as `ERROR` rather than silently as
  `UNKNOWN`. Note that `bmc` cannot in practice yield `SAFE`: the scan loop is
  `while(true)`, so incremental BMC never fully unwinds it and stops at the
  bound with `UNKNOWN` — which is what keeps the bounded-completeness caveat
  of §3.7 from being violated by the mapping.

The same `--ld-props` option is available on bare `esbmc` (e.g.
`esbmc program.ld --ld-props props.yaml --k-induction`), which is what the
regression tests drive directly.

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
theorem; it is the obligation that makes ESBMC-PLC a formal tool rather than a
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
- **Wall-clock timing accuracy.** The fixed-tick model (§3.3) proves properties in terms
  of scan counts, not wall-clock seconds. If the physical scan cycle time varies (jitter),
  the abstract tick count does not map directly to real time. Jitter analysis requires a
  separate real-time model and is out of scope for Tier 1.
- **Multi-task and interrupt-driven behaviour.** The theorem applies only to programs
  matching the single-task synchronous execution model (§3.3). Any program rejected with
  `UnsupportedConstruct(InterruptTask, tier=2)` is outside the theorem's domain.

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
include(FetchContent)
FetchContent_Declare(
  pugixml
  GIT_REPOSITORY https://github.com/zeux/pugixml.git
  GIT_TAG        v1.14
)
FetchContent_MakeAvailable(pugixml)

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

# irep2 and util are already linked transitively via the ESBMC build graph;
# explicit linkage follows the python-frontend pattern.
target_link_libraries(ldfrontend PUBLIC pugixml::static util irep2)
```

`tools/ld-verify/CMakeLists.txt` links `ldfrontend` and produces the `ld-verify` binary.

### 4.4 Dependencies

| Dependency | Role | Already in ESBMC? |
|---|---|---|
| pugixml | PLCopen XML DOM parsing | **No** — new dependency; bundled via `FetchContent_Declare(pugixml GIT_TAG v1.14)`. Single `.cpp` + header, no system dependency, links as `pugixml::static` on all platforms. Chosen over libxml2 to avoid static-linking fragility on macOS/Windows CI runners. |
| yaml-cpp | YAML property file parsing | **Yes** — already required (`src/util/CMakeLists.txt` links `yaml-cpp::yaml-cpp`; `util/yaml_parser.h` exposes the interface) |
| nlohmann/json | JSON report output | **Yes** — already used by the Python frontend |

---

## 5. Work Packages and Implementation Tasks

### WP1 — Formal Semantics & Requirements (Months 1–6)

| Task | Output | Milestone |
|---|---|---|
| T1.1 Systematic Literature Review (PRISMA) | SLR report | — |
| T1.2 SOS specification of IEC 61131-3 LD | `docs/safe-ld-sos-semantics.md` + LaTeX formalisation | M1 (Month 3): SOS spec v1 complete — **document written**; two-reviewer validation outstanding |
| T1.3 Property taxonomy & YAML format | `docs/safe-ld-property-format.md`; 20 synthetic validation programs | M2 (Month 6): property format validated |

**M1 gate:** SOS spec covers contacts, coils, TON/TOF/TP timers, CTU/CTD counters,
arithmetic FBs, and the cyclic scan model; validated against IEC 61131-3 §2 by two
independent reviewers.

**M2 gate:** YAML format applied to 20 synthetic programs representing all property kinds;
all 20 programs pass semantic review; spec reviewed against IEC 61508 §7.

### WP2 — ESBMC-PLC Tool Development (Months 4–12)

| Task | Subtasks | Milestone | Status |
|---|---|---|---|
| T2.1 Parser & Semantic Analyser | PLCopen XML parser; AST; type checker; SOS consistency check | M3 (Month 6): parser handles all WP1 SOS constructs | skeleton landed (#5280); extended with user-FB-body and REAL/analog parsing (#5620) |
| T2.2 GOTO IR Generator & Property Encoder | LdIR; `ld_converter` (irep2); YAML parser; property encoder (`code_assertt`) | M4 (Month 9): IR generator correct on all benchmark programs | boolean subset + timers/counters/`response` all lower and verify (#5289); ST→`codet` FB-body translator + numeric↔Boolean coercion (#5620); graphical resolver now models FB blocks, edge contacts, parallel-path OR and network feedback; all four benchmark verdicts validated (§10) |
| T2.3 ESBMC Integration & ld-verify | `ld_languaget`; CMake wiring; ld-verify CLI; JSON report | M5 (Month 12): end-to-end pipeline ready | `--ld-props` wired + JSON report (#5289); `ld-verify` runner implemented, driving `esbmc` (#5294); `--ld-fault-injection`, `--ld-sound-mode`, `--ld-scan-watchdog`/`--ld-scan-budget` driver flags added (#5294, #5620) |
| T2.4 Test Suite (TDD, >90% coverage) | Unit tests per component; integration tests; fault-injection tests | tracked per task; coverage measured with gcov | 3 unit suites + 37 driver regression tests (incl. fault-injection, user-FB, watchdog, REAL arithmetic) + 10 `ld-verify` runner tests, all run by CI; line-coverage target not yet measured |

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
| T4.4 Open-Source Release | ESBMC-PLC + ld-verify tagged release; TIA Portal + Codesys integration guides | M11 (Month 36): full open-source release |
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
| k-induction non-termination on timer-heavy programs | TON/TOF/TP timer state abstraction: `Q` modelled as nondet bool constrained by `__ESBMC_assume` to SOS timer invariants, reducing required induction depth to O(1); full concrete encoding retained as an option | **Not implemented.** `ld-verify --strategy` accepts only `bmc` and `k-induction`; `portfolio` and `abstract-timers` are rejected as unsupported (§3.6). No per-program timeout is applied |
| Solver timeout cascade in benchmark runs | Per-program timeout in `ld-verify` (default: 60 s); aggregate benchmark runner collects partial results and reports coverage fraction | `TIMEOUT` verdict treated as `UNKNOWN` in benchmark statistics; not counted as false positive or false negative |
| Unsupported LD constructs accumulation | Tiered support plan: **Tier 1** (WP2 scope) — contacts, coils, TON/TOF/TP, CTU/CTD, arithmetic FBs; **Tier 2** (post-project) — advanced FBs, structured text inline, arrays; **Tier 3** — vendor-specific extensions. Each unsupported construct emits a structured `UnsupportedConstruct(name, tier)` error, not a silent failure. | WP1 property taxonomy explicitly fixes the Tier 1 boundary; any Tier 2+ construct encountered in WP3 case studies is recorded as a known limitation in the paper |
| Incomplete PLCopen XML exports (missing FB declarations, partial networks) | Strict schema validation at parse time with diagnostic messages naming the missing element and the expected schema location | A library of known-valid exports from each vendor is maintained in `regression/ld/`; WP3 programs validated against the library before industrial use |
| Semantic drift across vendors (differing interpretations of IEC 61131-3 edge cases) | Vendor-specific SOS annotations in T1.2 document known divergences; regression tests cover each documented divergence | Divergences that affect verification results are flagged in `ld-verify` output with a `vendor-note` field |
| PDRA recruitment delay | Co-I bridges short-term | No implementation impact; timeline padded by 1 month per WP |
| Industrial programs not in PLCopen XML | Synthetic programs from published CSs; team has Codesys and TIA Portal access | WP3 CS programs collected in Month 10 |

---

## 8. Key Design Decisions

1. **Direct GOTO IR generation; no C intermediary.** ESBMC-PLC's `ld_converter` populates
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

---

## 10. Implementation Status

This section records what has actually landed against the plan above, so the
prose in §3 is not mistaken for delivered functionality.

### Landed

- **WP2 skeleton (#5280).** Front-end scaffolding: `ld_languaget`, PLCopen XML
  parser, type checker, LdIR + builder, `ld_converter`, YAML property parser,
  property encoder, `ld-verify` tool, CMake wiring behind `ENABLE_LD_FRONTEND`
  (default `OFF`), and unit tests.
- **Property pipeline (#5289).**
  - `--ld-props <file>` option on the `esbmc` driver; `ld_languaget::parse()`
    reads it (an explicit `set_props_path` from `ld-verify` still wins). Before
    this, the property file was never loaded and every program verified
    vacuously.
  - **READ_INPUTS** (§3.3): each `is_input` variable is re-sampled
    nondeterministically at the top of every scan iteration. Before this, inputs
    were frozen at their initial value, so verification was vacuous even with
    properties loaded.
  - **Typed arithmetic IR.** `plus_exprt`/`mult_exprt` leave their result type
    unset (the C frontend fills it in during its `adjust` pass; the LD frontend
    builds final IR directly and has none), so the timer/counter/`response`
    arithmetic previously migrated to a typeless `add2t` and aborted GOTO
    generation with `assert_arith_2ops_consistency` (`irep2_expr.cpp`). The
    arithmetic nodes are now built with an explicit result type, so TON/TOF/TP
    timers, CTU/CTD counters, and the `response` property lower and verify.
- **`ld-verify` runner + fault injection (#5294).** `LdVerifyRunner::run()`
  implemented end-to-end (see §3.6) with the verdict set
  `{SAFE, VIOLATION, INCOMPLETE, UNKNOWN, ERROR}`, `.xml`→`.ld` staging, and the
  JSON report. Fault injection is now exposed on the `esbmc` driver as
  `--ld-fault-injection` (previously only a `ld_converter` internal), with
  `fault_injection_safe`/`fault_injection_unsafe` regression tests.
- **Industrial benchmarks (#5322, #5360, #5427).** Real-world CONTROLLINO/OpenPLC
  programs added under `regression/ld/benchmarks/` (`stairs_light`,
  `water_control`), each with a README documenting its safety properties and
  expected verdict. (The NuXmv comparison referenced in #5427's title is not
  present in the merged tree; §3/§7 still list nuXmv as the WP3 comparator.)
- **User-defined FB bodies, REAL/analog types, scan-watchdog (#5620) — beyond
  Tier 1.** An ST→`codet` translator (`ir_gen/st_fb_translator.{h,cpp}`) compiles
  user-defined function-block bodies (assignments, `IF`, `WHILE`, comparisons,
  arithmetic), executed once per scan; FB output pins are wired back to program
  variables; `VarKind::REAL` (→ `double_type`) plus non-Boolean coils/contacts via
  numeric↔Boolean coercion; and an optional bounded scan-watchdog
  (`--ld-scan-watchdog` / `--ld-scan-budget`) that asserts each rung loop stays
  within budget. A `--ld-sound-mode` flag toggles the sound-vs-tolerant default
  for user-FB handling. Exercised by the `function_blocks_*`, `userfb_*`, and
  `arithmetic_div_unsafe` regression tests.
- **Graphical-LD soundness fixes.** The tc6_0201 resolver dropped any
  rail-to-coil path containing a function block, ignored the `edge` attribute
  on contacts, and let parallel paths to one coil overwrite each other instead
  of OR-ing — so `stairs_light` was verifying against a model with no timer, no
  edge detection and only its last branch. Blocks now resolve into synthesised
  `<instance>__<pin>` variables with presets converted to ticks via the declared
  task period, edges are sensed against a previous-scan shadow, parallel paths
  are OR-combined through a scratch accumulator, and a variable read and written
  in one network is snapshotted on entry per IEC 61131-3 §4.1.3. TOF and TP were
  also re-encoded so that `Q` starts false rather than reading an un-run timer's
  `ET` as an expired interval. PLCopen `<initialValue>` is now parsed; before,
  every declared preset silently read as zero.
- **Unmodellable rung paths are rejected, not dropped.** The graphical resolver
  used to skip any rail-to-coil path it could not model and, when a block's
  output drove a coil without a power-flow edge reaching it, emitted no rung and
  no diagnostic at all — leaving the coil at its initial value and passing
  properties over it vacuously. Both now raise
  `UnsupportedConstruct(name, tier=2)` per §7, naming the offending block.
- **Benchmark verdicts validated.** `conveyor_sequencing` and
  `emergency_shutdown` are wired as regression tests. The ESD violation proved
  to be a true positive — its reset rung does not gate on the manual trip being
  released, so a reset reopens the valve with the trip still asserted — and is
  pinned as `esd_manual_reset_fail`, with the corrected program as
  `emergency_shutdown_safe`. The conveyor's failure was a `response` property
  whose bound ignored a free `Stop_Button` input, plus the unparsed preset.
- **Regression suite `regression/ld/`** now holds **37 CTest cases** (guarded by
  `ENABLE_LD_FRONTEND`, with the `benchmarks/` dataset excluded from CTest —
  `regression/CMakeLists.txt`), covering all five property kinds plus
  fault-injection, user-FB, watchdog, REAL-arithmetic, and the `stairs_light` /
  `water_control` / `conveyor_sequencing` / `emergency_shutdown` industrial
  cases. The `ld-verify` runner has **10** further cases under the `ld-verify`
  label.
- **WP1 SOS specification (`docs/safe-ld-sos-semantics.md`).** The T1.2
  deliverable referenced by the M1 gate and by §3.7's theorem. Specifies the
  state space, the cyclic scan, contacts, coils, timers, counters, arithmetic
  blocks, graphical network resolution and the feedback rule, and gives the
  rule-to-GOTO-IR correspondence the theorem quantifies over.

### Working end-to-end

Contacts and coils (`--( )--`, `--( S )--`, `--( R )--`), TON/TOF/TP timers,
CTU/CTD counters, and all five property kinds — `mutual_exclusion`, `invariant`,
`absence`, `reachability`, `response` — plus user-defined function-block bodies,
REAL/analog process variables, and the optional scan-watchdog. These lower to
GOTO IR and verify under both k-induction and bounded BMC.

### CI status

`.github/workflows/plcplus-linux-binary.yml` carries two jobs and fires on pull
requests as well as pushes touching `src/ld-frontend/`, `tools/ld-verify/`,
`regression/ld/` or `unit/ld-frontend/`:

- `regression-ld` builds with `BUILD_TESTING=On` / `ENABLE_REGRESSION=On` and
  runs `regression/ld/` (37 cases), the `ld-verify` runner suite (10 cases) and
  the three LD unit binaries.
- `build-linux-amd64` builds the release binary, smoke-tests that it advertises
  `--ld-props`, and publishes it as an artifact.

The main PR matrix still does not enable `ENABLE_LD_FRONTEND`, so this workflow
is the only gate on the front-end.

### Known limitations / not yet validated

- **M1 review outstanding.** `docs/safe-ld-sos-semantics.md` now exists, but the
  M1 gate also requires validation by two independent reviewers against
  IEC 61131-3 §2. That review has not been carried out; §10 of the SOS document
  lists three specific points to raise in it (edge/polarity ordering, counter
  reset dominance, and the scope of the feedback rule).
- **WRITE_OUTPUTS** is not modelled as a distinct step; output coils are plain
  variable assignments (sufficient for the current property checks).
- **Timer/counter integer width — now saturating.** CTU/CTD saturate CV at the
  configured integer type's bound and TON bounds ET by PT, so neither wraps.
  Before this, `CV + 1` on a counter at INTmax was reachable undefined behaviour
  (`--overflow-check` reports `arithmetic overflow on add`) and the wrap dropped
  Q back to false, losing violations; `counter_saturate_at_max` pins it, and
  `counter_counts_fail` pins that the bound does not stop the counter counting.
  Whether IEC saturates CV at the type bound or at PV is recorded as open item 4
  for the M1 review in `docs/safe-ld-sos-semantics.md` §10 — the two agree on Q
  and differ only above the preset, and the type bound is the over-approximating
  (so non-hiding) choice.
- **Graphical path enumeration replaced by per-node accumulation.** The
  resolver used to enumerate every simple rail-to-sink path, so cost grew as 2^N
  in re-convergent parallel branches: 18 fully-connected 2-wide stages — just 36
  contacts — cost 112 s of GOTO-creation time, and a search bound was needed to
  reject such a program outright. It now computes
  `pf(n) = (OR over preds p of pf(p)) AND cond(n)` once per node, which is
  O(V+E), and the search bound is gone. The same 36-contact network resolves in
  0.13 s (`graphical_wide_network_fail`), and GOTO creation is linear in
  practice on fully re-convergent networks:

  | contacts | 150 | 450 | 1050 | 2100 |
  |---|---|---|---|---|
  | GOTO creation | 0.33 s | 0.37 s | 0.76 s | 1.50 s |

  This removes the resolver as the obstacle to the WP2 <5 s / 1000-rung
  criterion, but **does not meet it**. Measured end-to-end on ladder-shaped
  programs (each rung a short series chain, every third with a parallel branch),
  `--k-induction --unlimited-k-steps`, all verdicts SUCCESSFUL:

  | rungs | GOTO creation | end-to-end |
  |---|---|---|
  | 100 | 0.12 s | 2.6 s |
  | 250 | 0.26 s | 12.4 s |
  | 500 | 0.48 s | 48.0 s |
  | 1000 | 0.96 s | 190.9 s |

  GOTO creation is linear and is **0.5% of the runtime**; end-to-end grows
  roughly quadratically, so the criterion holds only to ~130 rungs and is missed
  by ~38x at 1000.

  The cost is **symex, not the solver**. At 500 rungs one symex pass takes 5.7 s
  for 12673 assignments, of which slicing then removes 12659 — 99.9% — leaving a
  single VCC that the solver discharges in 0.000 s. k-induction repeats that
  whole pass per step. So the lever is not the resolver or the solver but the
  work symex does on rungs the property never reads: slicing earlier, or
  restricting the scan body to the cone of influence of the properties, is what
  the criterion needs.
- **Arithmetic and unknown blocks on a rung path** are still not modelled —
  only timers and counters are resolved on graphical paths — but they are now
  a hard `UnsupportedConstruct(name, tier=2)` error rather than a dropped path,
  so a program using one is rejected instead of verifying over strictly less
  behaviour. Previously such a block was not even diagnosed: because step 3
  only makes `IN`/`CU`/`CD` into power-flow edges, a block like `GT` driving a
  coil got no incoming edge, `paths_to` returned nothing, and the coil was left
  unassigned — a `VERIFICATION SUCCESSFUL` verdict on a program whose rung had
  silently vanished (`graphical_unsupported_block_fail` pins this).
- **Counter reset from a contact chain** is diagnosed and left unconnected;
  only a reset pin wired to a variable is modelled.
- **Non-numeric presets used to terminate the process.** `literal_to_ticks`
  converted with `std::stoll` and relied on catching `std::invalid_argument`, but
  the catch did not fire: a block data pin wired to a named variable via
  `<inVariable>`, or an unparsable `<initialValue>`, aborted ESBMC with an
  uncaught exception. Both of the callers' fallback paths — the identifier
  reference in `resolve_data_pin` and the "unrecognised initial value" warning in
  `parse_var_decl` — were therefore unreachable. It now validates with `strtoll`
  and errno instead of converting and catching; `ld_preset_named_pin_safe` and
  `graphical_timer_path_fail` pin the two paths. Why the handler was skipped is
  not established, so the same convert-and-catch pattern elsewhere in ESBMC
  (~20 `std::sto*` call sites, several on user input) should not be assumed safe.

### Validation beyond the regression suite

`scripts/ld_resolver_oracle.py` is a randomised **structural** oracle for the
graphical resolver. It generates random PLCopen networks, derives each one's
power-flow formula from the ladder algebra independently of the resolver, and
asserts through ESBMC that the coil is equivalent to that formula (both
implications, so an undriven coil fails rather than passing vacuously). It is not
a differential test against a previous resolver, which would bless a bug present
in both. Two shapes are generated: `sp` (series-parallel, what a vendor tool
draws) and `dag` (layered, each node fed by a random subset of the previous
layer — not series-parallel, so it is what checks that path enumeration and the
per-node recurrence agree by distributivity).

This exists because the graphical corpus is only two real programs
(`stairs_light`, `water_control`), which is thin cover for changing the resolver.
Current state: **101 generated programs pass** (69 `sp` at depths 3–5, 32 `dag`
up to 4 layers x 3 wide). Run it before and after any resolver change:

```sh
python3 scripts/ld_resolver_oracle.py 30 4 sp        # series-parallel
python3 scripts/ld_resolver_oracle.py 20 3 dag 4 3   # re-convergent DAG
```

The oracle cannot be extended to stateful constructs, and this is a property of
the property language rather than an omission: the edge/feedback shadow update is
emitted *before* the scan-boundary assertion, so a previous-scan value is not
observable where properties are checked (`__edge_prev_a == a` there), and the
expression grammar has no temporal operators. So `q <=> f(inputs, prev)` is not
expressible.

Stateful constructs are therefore pinned by **discriminating reachability
tests** instead: a state that is reachable under the intended semantics and
provably unreachable under the plausible wrong one. `edge_rising_fail` /
`edge_falling_fail` assert that `a && !q` / `!a && q` are reachable, and
`edge_level_safe` proves both unreachable for a plain contact — so a resolver
that ignored the `edge` attribute again (as one did before #6378) flips all
three. `graphical_timer_path_fail` does the same for a timer on a graphical rung
path, which the enumerating resolver handles by cutting the path at the block and
resuming from its Q pin.

Set/Reset coils and multi-coil networks are pinned the same way:
`graphical_set_latch_fail` shows a set coil holding on with no path energised (an
output coil cannot); `graphical_set_reset_order_safe` proves the reset drawn
below the set wins within a scan, and **reversing the two `rightPowerRail`
connections flips it to FAILED**, so it pins sink emission order rather than
passing trivially; `graphical_multi_coil_safe` gives two coils different power
flows from one shared prefix, so a resolver handing both the same flow fails one
of the two invariants.

`graphical_feedback_snapshot_fail` closes the last one, the entry-snapshot rule
(§6.3 / IEC 61131-3 §4.1.3): `m` is written by the first rung and read by the
second, and the GOTO shows `m__prev = m` emitted before any rung, so the reader
sees the entry value and the scan in which `a` rises leaves `q` off. Reading `m`
immediately would make `q` track `a` and the state unreachable.

Note the textual-`<rung>` tests (`counter_*`, `function_blocks_*`, `userfb_*`)
bypass the graphical resolver entirely and are not cover for it.

**Together the oracle and these tests are the intended safety net for replacing
the resolver** (next increment 2): 101 generated programs for the combinational
algebra, plus one discriminating test per stateful construct — edge rising and
falling with a level complement, a timer on a rung path, set-coil latching, sink
emission order, multi-coil sub-network sharing, and the feedback snapshot — plus
the undriven-sink and unsupported-block rejections and the enumeration bound.

### Suggested next increments

1. Run the M1 review of `docs/safe-ld-sos-semantics.md` and close its §10 items.
2. Cut the symex cost that dominates end-to-end time (see the table above):
   slice against the properties' cone of influence before symex rather than
   after, so a 1000-rung program does not re-explore 12673 assignments per
   k-induction step to discharge one VCC.
3. Model arithmetic/unknown blocks on graphical rung paths, so the programs
   rejected as `UnsupportedConstruct` can be verified rather than refused.
