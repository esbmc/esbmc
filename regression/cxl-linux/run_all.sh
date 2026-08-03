#!/bin/bash
# Run every real-Linux harness and check it against its expected verdict.
#
# These harnesses are not registered with ctest: they compile actual kernel
# source and so need a configured kernel tree (see README.md). That makes them
# the least protected part of this work, which is why the expected verdicts
# belong in a script rather than only in a prose table.
#
# Usage:  ./run_all.sh            # uses the paths in run_esbmc.sh
# Exit:   0 if every harness matched, 1 otherwise.

set -u
cd "$(dirname "$0")" || exit 1
RUN=./run_esbmc.sh

# Real driver code has no bound on how much it costs to verify, and an
# unbounded run here took a 30 GiB machine to 3 GiB free before it was killed.
# A gate that can take the machine down with it is not a gate. Override with
# MEMLIMIT/TIMEOUT if you have the headroom.
LIMITS="--memlimit ${MEMLIMIT:-12g} --timeout ${TIMEOUT:-900}"

# harness | expected verdict | extra flags
CASES=(
  "harness_cdat_checksum.c|SUCCESSFUL|--unwind 12 --no-unwinding-assertions"
  "harness_cdat_checksum_fail.c|FAILED|--unwind 12 --no-unwinding-assertions"
  "harness_latency_nocontract.c|FAILED|"
  "harness_latency_contract.c|SUCCESSFUL|"
  "harness_dvsec_rr_decode.c|SUCCESSFUL|"
  "harness_dvsec_rr_decode_fail.c|FAILED|"
  # The --unwind bounds here are not tuning: cxl_hdm_decode_init() loops to
  # info->ranges, and __ESBMC_assume() constrains the value without bounding
  # the unwinding, so an unbounded run walks the loop forever. Each bound is
  # one past the harness's assumed maximum, and unwinding assertions stay on,
  # so the bound is proved rather than taken on trust.
  "harness_hdm_decode_init.c|SUCCESSFUL|--unwind 3 -DRANGES_BOUNDED"
  "harness_hdm_decode_init.c|FAILED|--unwind 5"
)

# The bounded harnesses must also hold with unwinding assertions on, otherwise
# the bound is an artefact of truncated unwinding rather than a real one.
SOUNDNESS=(
  "harness_cdat_checksum.c|SUCCESSFUL|--unwind 12"
)

fail=0
run_case() {
  local harness="$1" want="$2" flags="$3"
  local got
  got=$($RUN "$harness" $LIMITS $flags 2>&1 | grep -oE '^VERIFICATION (SUCCESSFUL|FAILED)' | tail -1)
  got=${got#VERIFICATION }
  if [ "$got" = "$want" ]; then
    printf '  ok       %-34s %-12s %s\n' "$harness" "$want" "$flags"
  else
    printf '  MISMATCH %-34s want %-11s got %s   %s\n' \
           "$harness" "$want" "${got:-<none>}" "$flags"
    fail=1
  fi
}

echo "== verdicts =="
for c in "${CASES[@]}"; do
  IFS='|' read -r h v f <<<"$c"
  run_case "$h" "$v" "$f"
done

echo "== unwinding soundness =="
for c in "${SOUNDNESS[@]}"; do
  IFS='|' read -r h v f <<<"$c"
  run_case "$h" "$v" "$f"
done

echo "== conversion only =="
if $RUN harness_core_pci.c --goto-functions-only >/dev/null 2>&1; then
  echo "  ok       harness_core_pci.c                 converts"
else
  echo "  MISMATCH harness_core_pci.c                 failed to convert"
  fail=1
fi

[ $fail -eq 0 ] && echo "all harnesses matched" || echo "MISMATCHES ABOVE"
exit $fail
