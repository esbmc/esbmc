/*  Boolector: Satisfiability Modulo Theories (SMT) solver.
 *
 *  Copyright (C) 2007-2021 by the authors listed in the AUTHORS file.
 *
 *  This file is part of Boolector.
 *  See COPYING for more information on using this software.
 */

#ifndef BTORTYPES_H_INCLUDED
#define BTORTYPES_H_INCLUDED

typedef struct Btor Btor;
typedef struct BtorNode BtorNode;

enum BtorSolverResult
{
  BTOR_RESULT_SAT     = 10,
  BTOR_RESULT_UNSAT   = 20,
  BTOR_RESULT_UNKNOWN = 0,
};

typedef enum BtorSolverResult BtorSolverResult;

/* public API types */
typedef struct BoolectorNode BoolectorNode;

typedef struct BoolectorAnonymous BoolectorAnonymous;

typedef BoolectorAnonymous* BoolectorSort;

/* --------------------------------------------------------------------- */
/* Boolector options                                                     */
/* --------------------------------------------------------------------- */

// clang-format off
enum BtorOption
{
  /* --------------------------------------------------------------------- */
  /*!
    **General Options:**
   */
  /* --------------------------------------------------------------------- */
  /*!
    * **BTOR_OPT_MODEL_GEN**

      | Enable (``value``: 1 or 2) or disable (``value``: 0) generation of a
        model for satisfiable instances.
      | There are two modes for model generation:

      * generate model for asserted expressions only (``value``: 1)
      * generate model for all expressions (``value``: 2)
  */
  BTOR_OPT_MODEL_GEN,

  /*!
    * **BTOR_OPT_INCREMENTAL**

      | Enable (``value``: 1) incremental mode.
      | Note that incremental usage turns off some optimization techniques.
        Disabling incremental usage is currently not supported.
  */
  BTOR_OPT_INCREMENTAL,

  /*!
    * **BTOR_OPT_INCREMENTAL_SMT1**

      | Set incremental mode for SMT-LIB v1 input.

      * BTOR_INCREMENTAL_SMT1_BASIC [default]:
        stop after first satisfiable formula
      * BTOR_INCREMENTAL_SMT1_CONTINUE:
        solve all formulas
  */
  BTOR_OPT_INCREMENTAL_SMT1,

  /*!
    * **BTOR_OPT_INPUT_FORMAT**

      | Force input file format.
      | If unspecified, Boolector automatically detects the input file format
        while parsing.

      * BTOR_INPUT_FORMAT_BTOR:
        `BTOR <http://fmv.jku.at/papers/BrummayerBiereLonsing-BPR08.pdf>`_ format
      * BTOR_INPUT_FORMAT_BTOR2:
        `BTOR2 <http://fmv.jku.at/papers/NiemetzPreinerWolfBiere-CAV18.pdf>`_ format
      * BTOR_INPUT_FORMAT_SMT1:
        `SMT-LIB v1 <http://smtlib.cs.uiowa.edu/papers/format-v1.2-r06.08.30.pdf>`_ format
      * BTOR_INPUT_FORMAT_SMT2:
        `SMT-LIB v2 <http://smtlib.cs.uiowa.edu/papers/smt-lib-reference-v2.0-r12.09.09.pdf>`_ format
  */

  BTOR_OPT_INPUT_FORMAT,
  /*!
    * **BTOR_OPT_OUTPUT_NUMBER_FORMAT**

      | Force output number format.

      * BTOR_OUTPUT_BASE_BIN [default]:
        binary
      * BTOR_OUTPUT_BASE_HEX:
        hexa-decimal
      * BTOR_OUTPUT_BASE_DEC:
        decimal
  */
  BTOR_OPT_OUTPUT_NUMBER_FORMAT,

  /*!
    * **BTOR_OPT_OUTPUT_FORMAT**

      | Force output file format (``value``: BTOR_: -1, `SMT-LIB v1`_: 1,
        `SMT-LIB v2`_: 2).

      * BTOR_OUTPUT_FORMAT_BTOR [default]:
        `BTOR`_ format
      * BTOR_OUTPUT_FORMAT_BTOR2:
        `BTOR2`_ format
      * BTOR_OUTPUT_FORMAT_SMT2:
        `SMT-LIB v2`_ format
      * BTOR_OUTPUT_FORMAT_AIGER_ASCII:
        `Aiger ascii format <http://fmv.jku.at/papers/BiereHeljankoWieringa-FMV-TR-11-2.pdf>`_
      * BTOR_OUTPUT_FORMAT_AIGER_BINARY:
        `Aiger binary format <http://fmv.jku.at/papers/BiereHeljankoWieringa-FMV-TR-11-2.pdf>`_
  */
  BTOR_OPT_OUTPUT_FORMAT,

  /*!
    * **BTOR_OPT_ENGINE**

      | Set solver engine.

      * BTOR_ENGINE_FUN [default]:
        the default engine for all combinations of QF_AUFBV, uses lemmas on
        demand for QF_AUFBV and eager bit-blasting for QF_BV
      * BTOR_ENGINE_SLS:
        the score-based local search QF_BV engine
      * BTOR_ENGINE_PROP:
        the propagation-based local search QF_BV engine
      * BTOR_ENGINE_AIGPROP:
        the propagation-based local search QF_BV engine that operates on the
        bit-blasted formula (the AIG layer)
      * BTOR_ENGINE_QUANT:
        the quantifier engine (BV only)
  */
  BTOR_OPT_ENGINE,

  /*!
    * **BTOR_OPT_SAT_ENGINE**

      | Set sat solver engine.
      | Available option values and default values depend on the sat solvers
        configured.

      * BTOR_SAT_ENGINE_CADICAL:
        `CaDiCaL <https://fmv.jku.at/cadical>`_
      * BTOR_SAT_ENGINE_CMS:
        `CryptoMiniSat <https://github.com/msoos/cryptominisat>`_
      * BTOR_SAT_ENGINE_LINGELING:
        `Lingeling <https://fmv.jku.at/lingeling>`_
      * BTOR_SAT_ENGINE_MINISAT:
        `MiniSat <https://github.com/niklasso/minisat>`_
      * BTOR_SAT_ENGINE_PICOSAT:
        `PicoSAT <http://fmv.jku.at/picosat/>`_
  */
  BTOR_OPT_SAT_ENGINE,

  /*!
    * **BTOR_OPT_AUTO_CLEANUP**

      Enable (``value``:1) or disable (``value``:0) auto cleanup of all
      references held on exit.
    */
  BTOR_OPT_AUTO_CLEANUP,

  /*!
    * **BTOR_OPT_PRETTY_PRINT**

      Enable (``value``: 1) or disable (``value``: 0) pretty printing when
      dumping.
  */
  BTOR_OPT_PRETTY_PRINT,

  /*!
    * **BTOR_OPT_EXIT_CODES**

      | Enable (``value``:1) or disable (``value``:0) the use of Boolector exit
        codes (BOOLECTOR_SAT, BOOLECTOR_UNSAT, BOOLECTOR_UNKNOWN - see
        :ref:`macros`).
      | If disabled, on exit Boolector returns 0 if success (sat or unsat), and
        1 in any other case.
  */
  BTOR_OPT_EXIT_CODES,

  /*!
    * **BTOR_OPT_SEED**

      | Set seed for Boolector's internal random number generator.
      | Boolector uses 0 by default.
  */
  BTOR_OPT_SEED,

  /*
    * **BTOR_OPT_VERBOSITY**

      Set the level of verbosity.
  */
  BTOR_OPT_VERBOSITY,

  /*
    * **BTOR_OPT_LOGLEVEL**

      Set the log level.
  */
  BTOR_OPT_LOGLEVEL,

  /* --------------------------------------------------------------------- */
  /*!
    **Simplifier Options:**
   */
  /* --------------------------------------------------------------------- */

  /*!
    * **BTOR_OPT_REWRITE_LEVEL**

      | Set the rewrite level (``value``: 0-3) of the rewriting engine.
      | Boolector uses rewrite level 3 by default, rewrite levels are
        classified as follows:

      * 0: no rewriting
      * 1: term level rewriting
      * 2: more simplification techniques
      * 3: full rewriting/simplification

      | Do not alter the rewrite level of the rewriting engine after creating
        expressions.
  */
  BTOR_OPT_REWRITE_LEVEL,

  /*!
    * **BTOR_OPT_SKELETON_PREPROC**

      Enable (``value``: 1) or disable (``value``: 0) skeleton  preprocessing
      during simplification.
  */
  BTOR_OPT_SKELETON_PREPROC,

  /*!
    * **BTOR_OPT_ACKERMANN**

      Enable (``value``: 1) or disable (``value``: 0) the eager addition of
      Ackermann constraints for function applications.
  */
  BTOR_OPT_ACKERMANN,

  /*!
    * **BTOR_OPT_BETA_REDUCE**

      Enable (``value``: 1) or disable (``value``: 0) the eager elimination of
      lambda expressions via beta reduction.
  */
  BTOR_OPT_BETA_REDUCE,

  /*!
    * **BTOR_OPT_ELIMINATE_SLICES**

      Enable (``value``: 1) or disable (``value``: 0) slice elimination on bit
      vector variables.
  */
  BTOR_OPT_ELIMINATE_SLICES,

  /*!
    * **BTOR_OPT_VAR_SUBST**

      Enable (``value``: 1) or disable (``value``: 0) variable substitution
      during simplification.
  */
  BTOR_OPT_VAR_SUBST,

  /*!
    * **BTOR_OPT_UCOPT**

      Enable (``value``: 1) or disable (``value``: 0) unconstrained
      optimization.
  */
  BTOR_OPT_UCOPT,

  /*!
    * **BTOR_OPT_MERGE_LAMBDAS**

      Enable (``value``: 1) or disable (``value``: 0) merging of lambda
      expressions.
  */
  BTOR_OPT_MERGE_LAMBDAS,

  /*!
    * **BTOR_OPT_EXTRACT_LAMBDAS**

      Enable (``value``: 1) or disable (``value``: 0) extraction of common
      array patterns as lambda terms.
  */
  BTOR_OPT_EXTRACT_LAMBDAS,

  /*!
    * **BTOR_OPT_NORMALIZE**

      Enable (``value``: 1) or disable (``value``: 0) normalization of
      addition, multiplication and bit-wise and.
  */
  BTOR_OPT_NORMALIZE,

  /*!
    * **BTOR_OPT_NORMALIZE_ADD**

      Enable (``value``: 1) or disable (``value``: 0) normalization of
      addition.
  */
  BTOR_OPT_NORMALIZE_ADD,

  /* --------------------------------------------------------------------- */
  /*!
    **Fun Engine Options:**
   */
  /* --------------------------------------------------------------------- */

  /*!
    * **BTOR_OPT_FUN_PREPROP**

      Enable (``value``: 1) or disable (``value``: 0) prop engine as
      preprocessing step within sequential portfolio setting.
   */
  BTOR_OPT_FUN_PREPROP,

  /*!
    * **BTOR_OPT_FUN_PRESLS**

      Enable (``value``: 1) or disable (``value``: 0) sls engine as
      preprocessing step within sequential portfolio setting.
   */
  BTOR_OPT_FUN_PRESLS,

  /*!
    * **BTOR_OPT_FUN_DUAL_PROP**

      Enable (``value``: 1) or disable (``value``: 0) dual propagation
      optimization.
  */
  BTOR_OPT_FUN_DUAL_PROP,

  /*!
    * **BTOR_OPT_FUN_DUAL_PROP_QSORT**

      | Set order in which inputs are assumed in dual propagation clone.

      * BTOR_DP_QSORT_JUST [default]:
        order by score, highest score first
      * BTOR_DP_QSORT_ASC:
        order by input id, ascending
      * BTOR_DP_QSORT_DESC:
        order by input id, descending
  */
  BTOR_OPT_FUN_DUAL_PROP_QSORT,

  /*!
    * **BTOR_OPT_FUN_JUST**

      Enable (``value``: 1) or disable (``value``: 0) justification
      optimization.
  */
  BTOR_OPT_FUN_JUST,

  /*!
    * **BTOR_OPT_FUN_JUST_HEURISTIC**

      | Set heuristic that determines path selection for justification
        optimization.

      * BTOR_JUST_HEUR_BRANCH_MIN_APP [default]:
        choose branch with minimum number of applies
      * BTOR_JUST_HEUR_BRANCH_MIN_DEP:
        choose branch with minimum depth
      * BTOR_JUST_HEUR_LEFT:
        always choose left branch
  */
  BTOR_OPT_FUN_JUST_HEURISTIC,

  /*!
    * **BTOR_OPT_FUN_LAZY_SYNTHESIZE**

      Enable (``value``: 1) or disable (``value``: 0) lazy synthesis of bit
      vector expressions.
  */
  BTOR_OPT_FUN_LAZY_SYNTHESIZE,

  /*!
    * **BTOR_OPT_FUN_EAGER_LEMMAS**

      | Select mode for eager generation lemmas.

      * BTOR_FUN_EAGER_LEMMAS_NONE:
        do not generate lemmas eagerly (generate one single lemma per
        refinement iteration)
      * BTOR_FUN_EAGER_LEMMAS_CONF:
        only generate lemmas eagerly until the first conflict dependent on
        another conflict is found
      * BTOR_FUN_EAGER_LEMMAS_ALL:
        in each refinement iteration, generate lemmas for all conflicts
  */
  BTOR_OPT_FUN_EAGER_LEMMAS,

  BTOR_OPT_FUN_STORE_LAMBDAS,

  /*!
    * **BTOR_OPT_PRINT_DIMACS**

      Enable (``value``: 1) or disable (``value``: 0) DIMACS printer.

      When enabled Boolector will record the CNF sent to the SAT solver and
      prints it to stdout.
   */
  BTOR_OPT_PRINT_DIMACS,


  /* --------------------------------------------------------------------- */
  /*!
    **SLS Engine Options:**
   */
  /* --------------------------------------------------------------------- */

  /*!
    * **BTOR_OPT_SLS_NFIPS**
      Set the number of bit flips used as a limit for the sls engine. Disabled
      if 0.
   */
  BTOR_OPT_SLS_NFLIPS,

  /*!
    * **BTOR_OPT_SLS_STRATEGY**

      | Select move strategy for SLS engine.

      * BTOR_SLS_STRAT_BEST_MOVE:
        always choose best score improving move
      * BTOR_SLS_STRAT_RAND_WALK:
        always choose random walk weighted by score
      * BTOR_SLS_STRAT_FIRST_BEST_MOVE [default]:
        always choose first best move (no matter if any other move is better)
      * BTOR_SLS_STRAT_BEST_SAME_MOVE:
        determine move as best move even if its score is not better but the
        same as the score of the previous best move
      * BTOR_SLS_STRAT_ALWAYS_PROP:
        always choose propagation move (and recover with SLS move in case of
        conflict)
  */
  BTOR_OPT_SLS_STRATEGY,

  /*!
    * **BTOR_OPT_SLS_JUST**

      Enable (``value``: 1) or disable (``value``: 0) justification based path
      selection during candidate selection.
  */
  BTOR_OPT_SLS_JUST,

  /*!
    * **BTOR_OPT_SLS_MOVE_GW**

      Enable (``value``: 1) or disable (``value``: 0) group-wise moves, where
      rather than changing the assignment of one single candidate variable, all
      candidate variables are set at the same time (using the same strategy).
  */
  BTOR_OPT_SLS_MOVE_GW,

  /*!
    * **BTOR_OPT_SLS_MOVE_RANGE**

      Enable (``value``: 1) or disable (``value``: 0) range-wise bit-flip
      moves, where the bits within all ranges from 2 to the bit width (starting
      from the LSB) are flipped at once.
  */
  BTOR_OPT_SLS_MOVE_RANGE,

  /*!
    * **BTOR_OPT_SLS_MOVE_SEGMENT**

      Enable (``value``: 1) or disable (``value``: 0) segment-wise bit-flip
      moves, where the bits within segments of multiples of 2 are flipped at
      once.
  */
  BTOR_OPT_SLS_MOVE_SEGMENT,

  /*!
    * **BTOR_OPT_SLS_MOVE_RAND_WALK**

      Enable (``value``: 1) or disable (``value``: 0) random walk moves, where
      one out of all possible neighbors is randomly selected (with given
      probability, see BTOR_OPT_SLS_PROB_MOVE_RAND_WALK) for a randomly
      selected candidate variable.
  */
  BTOR_OPT_SLS_MOVE_RAND_WALK,

  /*!
    * **BTOR_OPT_SLS_PROB_MOVE_RAND_WALK**

      Set the probability with which a random walk is chosen if random walks
      are enabled (see BTOR_OPT_SLS_MOVE_RAND_WALK).
  */
  BTOR_OPT_SLS_PROB_MOVE_RAND_WALK,

  /*!
    * **BTOR_OPT_SLS_MOVE_RAND_ALL**

      Enable (``value``: 1) or disable (``value``: 0) the randomization of all
      candidate variables (rather than a single randomly selected candidate
      variable) in case no best move has been found.
  */
  BTOR_OPT_SLS_MOVE_RAND_ALL,

  /*!
    * **BTOR_OPT_SLS_MOVE_RAND_RANGE**

      Enable (``value``: 1) or disable (``value``: 0) the randomization of bit
      ranges (rather than all bits) of a candidate variable(s) to be randomized
      in case no best move has been found.
  */
  BTOR_OPT_SLS_MOVE_RAND_RANGE,

  /*!
    * **BTOR_OPT_SLS_MOVE_PROP**

      Enable (``value``: 1) or disable (``value``: 0) propagation moves (chosen
      with a given ratio of number of propagation moves to number of regular
      SLS moves, see BTOR_OPT_SLS_MOVE_PROP_N_PROP and
      BTOR_OPT_SLS_MOVE_PROP_N_SLS).
  */
  BTOR_OPT_SLS_MOVE_PROP,

  /*!
    * **BTOR_OPT_SLS_MOVE_PROP_N_PROP**

      Set the number of propagation moves to be performed when propagation
      moves are enabled (propagation moves are chosen with a ratio of
      propagation moves to regular SLS moves, see BTOR_OPT_SLS_MOVE_PROP and
      BTOR_OPT_SLS_MOVE_PROP_N_SLS).
  */
  BTOR_OPT_SLS_MOVE_PROP_N_PROP,

  /*!
    * **BTOR_OPT_SLS_MOVE_PROP_N_SLS**

      Set the number of regular SLS moves to be performed when propagation
      moves are enabled (propagation moves are chosen with a ratio of
      propagation moves to regular SLS moves, see BTOR_OPT_SLS_MOVE_PROP and
      BTOR_OPT_SLS_MOVE_PROP_N_PROP).
  */
  BTOR_OPT_SLS_MOVE_PROP_N_SLS,

  /*!
    * **BTOR_OPT_SLS_MOVE_PROP_FORCE_RW**

      Enable (``value``: 1) or disable (``value``: 0) that random walks are
      forcibly chosen as recovery moves in case of conflicts when a propagation
      move is performed (rather than performing a regular SLS move).
  */
  BTOR_OPT_SLS_MOVE_PROP_FORCE_RW,

  /*!
    * **BTOR_OPT_SLS_MOVE_INC_MOVE_TEST**

      Enable (``value``: 1) or disable (``value``: 0) that during best move
      selection, if the current candidate variable with a previous neighbor
      yields the currently best score, this neighbor assignment is used as a
      base for further neighborhood exploration (rather than its current
      assignment).
  */
  BTOR_OPT_SLS_MOVE_INC_MOVE_TEST,

  /*!
    * **BTOR_OPT_SLS_MOVE_RESTARTS**

      Enable (``value``: 1) or disable (``value``: 0) restarts.
  */
  BTOR_OPT_SLS_USE_RESTARTS,

  /*!
    * **BTOR_OPT_SLS_MOVE_RESTARTS**

      | Enable (``value``: 1) or disable (``value``: 0) heuristic (bandit
        scheme) for selecting root constraints.
      | If disabled, candidate root constraints are selected randomly.
  */
  BTOR_OPT_SLS_USE_BANDIT,

  /* --------------------------------------------------------------------- */
  /*!
    **Prop Engine Options**:
   */
  /* --------------------------------------------------------------------- */

  /*!
    * **BTOR_OPT_PROP_NPROPS**

      Set the number of propagation (steps) used as a limit for the propagation
      engine. Disabled if 0.
   */
  BTOR_OPT_PROP_NPROPS,

  /*!
    * **BTOR_OPT_PROP_USE_RESTARTS**

      Enable (``value``: 1) or disable (``value``: 0) restarts.
  */
  BTOR_OPT_PROP_USE_RESTARTS,

  /*!
    * **BTOR_OPT_PROP_USE_RESTARTS**

      | Enable (``value``: 1) or disable (``value``: 0) heuristic (bandit
        scheme) for selecting root constraints.
      | If enabled, root constraint selection via bandit scheme is based on a
        scoring scheme similar to the one employed in the SLS engine.
      | If disabled, candidate root constraints are selected randomly.
  */
  BTOR_OPT_PROP_USE_BANDIT,

  /*!
    * **BTOR_OPT_PROP_PATH_SEL**

      Select mode for path selection.

      * BTOR_PROP_PATH_SEL_CONTROLLING:
        select path based on controlling inputs
      * BTOR_PROP_PATH_SEL_ESSENTIAL [default]:
        select path based on essential inputs
      * BTOR_PROP_PATH_SEL_RANDOM:
        select path based on random inputs
  */
  BTOR_OPT_PROP_PATH_SEL,

  /*!
    * **BTOR_OPT_PROP_PROB_USE_INV_VALUE**

     Set probabiality with which to choose inverse values over consistent
     values.
  */
  BTOR_OPT_PROP_PROB_USE_INV_VALUE,

  /*!
    * **BTOR_OPT_PROP_PROB_FLIP_COND**

     Set probability with which to select the path to the condition (in case of
     an if-then-else operation) rather than the enabled branch during down
     propagation.
  */
  BTOR_OPT_PROP_PROB_FLIP_COND,

  /*!
    * **BTOR_OPT_PROP_PROB_FLIP_COND_CONST**

     Set probbiality with which to select the path to the condition (in case of
     an if-then-else operation) rather than the enabled branch during down
     propagation if either of the 'then' or 'else' branch is constant.
  */
  BTOR_OPT_PROP_PROB_FLIP_COND_CONST,

  /*!
    * **BTOR_OPT_PROP_FLIP_COND_CONST_DELTA**

     Set delta by which BTOR_OPT_PROP_PROB_FLIP_COND_CONST is decreased or
     increased after a limit BTOR_OPT_PROP_FLIP_COND_CONST_NPATHSEL is reached.
  */
  BTOR_OPT_PROP_FLIP_COND_CONST_DELTA,

  /*!
    * **BTOR_OPT_PROP_FLIP_COND_CONST_NPATHSEL**

     Set the limit for how often the path to the condition (in case of an
     if-then-else operation) may be selected bevor
     BTOR_OPT_PROP_PROB_FLIP_COND_CONST is decreased or increased by
     BTOR_OPT_PROP_PROB_FLIP_COND_CONST_DELTA.
  */
  BTOR_OPT_PROP_FLIP_COND_CONST_NPATHSEL,

  /*!
    * **BTOR_OPT_PROP_PROB_SLICE_KEEP_DC**

      Set probability with which to keep the current value of the don't care
      bits of a slice operation (rather than fully randomizing all of them)
      when selecting an inverse or consistent value.
   */
  BTOR_OPT_PROP_PROB_SLICE_KEEP_DC,

  /*!
    * **BTOR_OPT_PROP_PROB_CONC_FLIP**

     Set probability with which to use the corresponing slice of current
     assignment with max. one of its bits flipped (rather than using the
     corresponding slice of the down propagated assignment) as result of
     consistent value selection for concats.
  */
  BTOR_OPT_PROP_PROB_CONC_FLIP,

  /*!
    * **BTOR_OPT_PROP_PROB_SLICE_FLIP**

     Set probability with which to use the current assignment of the operand of
     a slice operation with one of the don't care bits flipped (rather than
     fully randomizing all of them, both for inverse and consistent value
     selection) if their current assignment is not kept (see
     BTOR_OPT_PROP_PROB_SLICE_KEEP_DC).
  */
  BTOR_OPT_PROP_PROB_SLICE_FLIP,

  /*!
    * **BTOR_OPT_PROP_PROB_EQ_FLIP**

     Set probability with which the current assignment of the selected node
     with one of its bits flipped (rather than a fully randomized bit-vector)
     is down-propagated in case of an inequality (both for inverse and
     consistent value selection).
  */
  BTOR_OPT_PROP_PROB_EQ_FLIP,

  /*!
    * **BTOR_OPT_PROP_PROB_AND_FLIP**

     Set probability with which the current assignment of the don't care bits
     of the selected node with max. one of its bits flipped (rather than fully
     randomizing all of them) in case of an and operation (for both inverse and
     consistent value selection).

  */
  BTOR_OPT_PROP_PROB_AND_FLIP,

  /*!
    * **BTOR_OPT_PROP_NO_MOVE_ON_CONFLICT**

      | Do not perform a propagation move when running into a conflict during
        inverse computation.
      | (This is the default behavior for the SLS engine when propagation moves
        are enabled, where a conflict triggers a recovery by means of a regular
        SLS move.)
    */
  BTOR_OPT_PROP_NO_MOVE_ON_CONFLICT,

  /* --------------------------------------------------------------------- */
  /*!
    **AIGProp Engine Options**:
   */
  /* --------------------------------------------------------------------- */

  /*!
    * **BTOR_OPT_AIGPROP_USE_RESTARTS**

      Enable (``value``: 1) or disable (``value``: 0) restarts.
  */
  BTOR_OPT_AIGPROP_USE_RESTARTS,

  /*!
    * **BTOR_OPT_AIGPROP_USE_RESTARTS**

      | Enable (``value``: 1) or disable (``value``: 0) heuristic (bandit
        scheme) for selecting root constraints.
      | If enabled, root constraint selection via bandit scheme is based on a
        scoring scheme similar to the one employed in the SLS engine.
      | If disabled, candidate root constraints are selected randomly.
  */
  BTOR_OPT_AIGPROP_USE_BANDIT,

  /* QUANT engine ------------------------------------------------------- */
  /*!
    * **BTOR_OPT_QUANT_SYNTH**

     Select synthesis mode for Skolem functions.

     * BTOR_QUANT_SYNTH_NONE:
       do not synthesize skolem functions (use model values for instantiation)
     * BTOR_QUANT_SYNTH_EL:
       use enumerative learning to synthesize skolem functions
     * BTOR_QUANT_SYNTH_ELMC:
       use enumerative learning modulo the predicates in the cone of influence
       of the existential variables to synthesize skolem functions
     * BTOR_QUANT_SYNTH_EL_ELMC:
       chain BTOR_QUANT_SYNTH_EL and BTOR_QUANT_SYNTH_ELMC approaches to
       synthesize skolem functions
     * BTOR_QUANT_SYNTH_ELMR:
       use enumerative learning modulo the given root constraints to synthesize
       skolem functions
  */
  BTOR_OPT_QUANT_SYNTH,

  /*!
    * **BTOR_OPT_QUANT_DUAL_SOLVER**

      Enable (``value``: 1) or disable (``value``: 0) solving the dual
      (negated) version of the quantified bit-vector formula.
   */
  BTOR_OPT_QUANT_DUAL_SOLVER,

  /*!
    * **BTOR_OPT_QUANT_SYNTH_LIMIT**

      Set the limit of enumerated expressions for the enumerative learning
      synthesis algorithm.
   */
  BTOR_OPT_QUANT_SYNTH_LIMIT,

  /*!
    * **BTOR_OPT_SYNTH_QI**

      Enable (``value``: 1) or disable (``value``: 0) generalization of
      quantifier instantiations via enumerative learning.
   */
  BTOR_OPT_QUANT_SYNTH_QI,

  /*!
    * **BTOR_OPT_QUANT_DER**

      Enable (``value``: 1) or disable (``value``: 0) destructive equality
      resolution simplification.
   */
  BTOR_OPT_QUANT_DER,

  /*!
    * **BTOR_OPT_QUANT_CER**

      Enable (``value``: 1) or disable (``value``: 0) constructive equality
      resolution simplification.
   */
  BTOR_OPT_QUANT_CER,

  /*!
    * **BTOR_OPT_QUANT_MINISCOPE**

      Enable (``value``: 1) or disable (``value``: 0) miniscoping.
   */
  BTOR_OPT_QUANT_MINISCOPE,

  /* internal options --------------------------------------------------- */

  BTOR_OPT_SORT_EXP,
  BTOR_OPT_SORT_AIG,
  BTOR_OPT_SORT_AIGVEC,
  BTOR_OPT_AUTO_CLEANUP_INTERNAL,
  BTOR_OPT_SIMPLIFY_CONSTRAINTS,
  BTOR_OPT_CHK_FAILED_ASSUMPTIONS,
  BTOR_OPT_CHK_MODEL,
  BTOR_OPT_CHK_UNCONSTRAINED,
  BTOR_OPT_PARSE_INTERACTIVE,
  BTOR_OPT_SAT_ENGINE_LGL_FORK,
  BTOR_OPT_SAT_ENGINE_CADICAL_FREEZE,
  BTOR_OPT_SAT_ENGINE_N_THREADS,
  BTOR_OPT_SIMP_NORMAMLIZE_ADDERS,
  BTOR_OPT_DECLSORT_BV_WIDTH,
  BTOR_OPT_QUANT_SYNTH_ITE_COMPLETE,
  BTOR_OPT_QUANT_FIXSYNTH,
  BTOR_OPT_RW_ZERO_LOWER_SLICE,
  BTOR_OPT_NONDESTR_SUBST,
  /* this MUST be the last entry! */
  BTOR_OPT_NUM_OPTS,
};
// clang-format on

typedef enum BtorOption BtorOption;

/* --------------------------------------------------------------------- */
/* Boolector option values                                               */
/* --------------------------------------------------------------------- */

/* Note: enums with NONE values should start with NONE = 0. If there is no NONE
 * value the enum range should start with 1. This allows us to determine if an
 * option is set by checking if it is > 0. */

enum BtorOptSatEngine
{
  BTOR_SAT_ENGINE_LINGELING,
  BTOR_SAT_ENGINE_PICOSAT,
  BTOR_SAT_ENGINE_MINISAT,
  BTOR_SAT_ENGINE_CADICAL,
  BTOR_SAT_ENGINE_CMS,
};
typedef enum BtorOptSatEngine BtorOptSatEngine;

enum BtorOptEngine
{
  BTOR_ENGINE_FUN = 1,
  BTOR_ENGINE_SLS,
  BTOR_ENGINE_PROP,
  BTOR_ENGINE_AIGPROP,
  BTOR_ENGINE_QUANT,
};
typedef enum BtorOptEngine BtorOptEngine;

enum BtorOptInputFormat
{
  BTOR_INPUT_FORMAT_NONE,
  BTOR_INPUT_FORMAT_BTOR,
  BTOR_INPUT_FORMAT_BTOR2,
  BTOR_INPUT_FORMAT_SMT1,
  BTOR_INPUT_FORMAT_SMT2,
};
typedef enum BtorOptInputFormat BtorOptInputFormat;

enum BtorOptOutputBase
{
  BTOR_OUTPUT_BASE_BIN = 1,
  BTOR_OUTPUT_BASE_HEX,
  BTOR_OUTPUT_BASE_DEC,
};
typedef enum BtorOptOutputBase BtorOptOutputBase;

enum BtorOptOutputFormat
{
  BTOR_OUTPUT_FORMAT_NONE,
  BTOR_OUTPUT_FORMAT_BTOR = 1,
  //  BTOR_OUTPUT_FORMAT_BTOR2,
  BTOR_OUTPUT_FORMAT_SMT2,
  BTOR_OUTPUT_FORMAT_AIGER_ASCII,
  BTOR_OUTPUT_FORMAT_AIGER_BINARY,
};
typedef enum BtorOptOutputFormat BtorOptOutputFormat;

enum BtorOptDPQsort
{
  BTOR_DP_QSORT_JUST = 1,
  BTOR_DP_QSORT_ASC,
  BTOR_DP_QSORT_DESC,
};
typedef enum BtorOptDPQsort BtorOptDPQsort;

enum BtorOptJustHeur
{
  BTOR_JUST_HEUR_BRANCH_LEFT = 1,
  BTOR_JUST_HEUR_BRANCH_MIN_APP,
  BTOR_JUST_HEUR_BRANCH_MIN_DEP,
};
typedef enum BtorOptJustHeur BtorOptJustHeur;

enum BtorOptSLSStrat
{
  BTOR_SLS_STRAT_BEST_MOVE = 1,
  BTOR_SLS_STRAT_RAND_WALK,
  BTOR_SLS_STRAT_FIRST_BEST_MOVE,
  BTOR_SLS_STRAT_BEST_SAME_MOVE,
  BTOR_SLS_STRAT_ALWAYS_PROP,
};
typedef enum BtorOptSLSStrat BtorOptSLSStrat;

enum BtorOptPropPathSel
{
  BTOR_PROP_PATH_SEL_CONTROLLING = 1,
  BTOR_PROP_PATH_SEL_ESSENTIAL,
  BTOR_PROP_PATH_SEL_RANDOM,
};
typedef enum BtorOptPropPathSel BtorOptPropPathSel;

enum BtorOptQuantSynth
{
  BTOR_QUANT_SYNTH_NONE,
  BTOR_QUANT_SYNTH_EL,
  BTOR_QUANT_SYNTH_ELMC,
  BTOR_QUANT_SYNTH_EL_ELMC,
  BTOR_QUANT_SYNTH_ELMR,
};
typedef enum BtorOptQuantSynth BtorOptQuantSynt;

enum BtorOptFunEagerLemmas
{
  BTOR_FUN_EAGER_LEMMAS_NONE,
  BTOR_FUN_EAGER_LEMMAS_CONF,
  BTOR_FUN_EAGER_LEMMAS_ALL,
};
typedef enum BtorOptFunEagerLemmas BtorOptFunEagerLemmas;

enum BtorOptIncrementalSMT1
{
  BTOR_INCREMENTAL_SMT1_BASIC = 1,
  BTOR_INCREMENTAL_SMT1_CONTINUE,
};
typedef enum BtorOptIncrementalSMT1 BtorOptIncrementalSMT1;

enum BtorOptBetaReduceMode
{
  BTOR_BETA_REDUCE_NONE,
  BTOR_BETA_REDUCE_FUN,
  BTOR_BETA_REDUCE_ALL,
};
typedef enum BtorOptBetaReduceMode BtorOptBetaReduceMode;

/* --------------------------------------------------------------------- */

/* Callback function to be executed on abort, primarily intended to be used for
 * plugging in exception handling. */
struct BtorAbortCallback
{
  void (*abort_fun) (const char* msg);
  void *cb_fun;   /* abort callback function */
};
typedef struct BtorAbortCallback BtorAbortCallback;

extern BtorAbortCallback btor_abort_callback;

#endif
