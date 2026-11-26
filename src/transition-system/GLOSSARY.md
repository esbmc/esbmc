# Transition System

Turning a C program's main loop into a transition system (states, an initial
condition, a step, and bad states) so that model checkers such as PDR, or
external BTOR2 tools, can verify it.

## Language

**Extractable**:
A program that the transition-system check accepts.
_Avoid_: matched, supported, accepted program

**Prefix**:
The code that runs once, before the loop is first entered.
_Avoid_: preamble, init code

**Step**:
One execution of the loop from its head, either around the back edge or out
through the exit.
_Avoid_: body (when the exit path is meant), iteration

**State**:
The variables carried from one step to the next: the loop's havocked variables
that the properties depend on.
_Avoid_: registers, latches

**Havoc**:
Giving a variable an arbitrary value at the loop head, so that the head stands
for every iteration rather than the first one.

**Back-edge guard**:
The condition under which a step returns to the loop head.

**Exit**:
A step that does not take the back edge and leaves the loop.

**Entry condition**:
The loop condition that k-induction assumes right after the havoc. It holds
only in steps that stay in the loop, so it is not an invariant.

**Invariant**:
A fact that holds for every state at the loop head and that only the
inductive step may assume, such as an interval-analysis bound.

**Suffix**:
The code that runs after the loop is left by its exit.
_Avoid_: tail, post-loop code, epilogue

## Several loops

**Flattening**:
Rewriting a function with several loops into one loop whose steps each run
one block.

**Cut point**:
A point where flattening splits the code: a loop head, or the target of a jump
that enters a loop, or the code after the first loop head, from outside.

**Block**:
The loop-free code a step runs, from one cut point to the next.

**Location**:
The state variable naming the cut point the next step starts at.
_Avoid_: pc, program counter, mode
