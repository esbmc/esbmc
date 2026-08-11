#!/bin/bash
# Status of the 12 wide-format sqrt oracle validations (6 formats x 2 solvers),
# launched with no timeout. Safe to run repeatedly.
#
#   ./check_widesqrt.sh
#
# A format is PROVED once either solver returns SUCCESSFUL -- the two are
# racing the same query, so the first to finish settles it.
R=${1:-/home/mgadelha/.claude/jobs/0332a978/tmp/widesqrt}
[ -d "$R" ] || { echo "no run directory at $R"; exit 1; }

printf '%-8s %-24s %-24s %s\n' FORMAT BITWUZLA Z3 SETTLED
printf '%.0s-' {1..76}; echo

live=0
for f in lr ulr k uk lk ulk; do
  line=""
  settled=""
  for slv in bitwuzla z3; do
    log=$R/${f}_${slv}.log
    if [ ! -f "$log" ]; then
      st="(no log)"
    else
      v=$(grep -m1 -E '^VERIFICATION' "$log" 2>/dev/null)
      t=$(grep -m1 -oE '@@ELAPSED [0-9.]+' "$log" 2>/dev/null | awk '{print $2}')
      if [ -n "$v" ]; then
        st="${v#VERIFICATION } ${t:+(${t}s)}"
        [ "${v#VERIFICATION }" = "SUCCESSFUL" ] && settled="PROVED by $slv"
        [ "${v#VERIFICATION }" = "FAILED" ] && settled="${settled:-ORACLE BUG? ($slv)}"
      elif pgrep -f "esbmc oracle_sqrt_${f}.c$( [ "$slv" = z3 ] && echo ' --z3')" >/dev/null 2>&1; then
        st="running"; live=$((live+1))
      else
        st="died/no verdict"
      fi
    fi
    line="$line$(printf '%-24s' "$st")"
  done
  printf '%-8s %s%s\n' "$f" "$line" "${settled:-still open}"
done

echo
echo "live esbmc processes: $(pgrep -cx esbmc 2>/dev/null || echo 0)"
echo "uptime/load: $(uptime | sed 's/.*load average/load average/')"
echo
echo "Formats: lr=s0.31 ulr=u0.32 k=s16.15 uk=u16.16 lk=s32.31 ulk=u32.32"
echo "A SUCCESSFUL verdict upgrades that format from anchor-validated to"
echo "proved over all inputs. A FAILED verdict would mean the bracket or the"
echo "oracle is wrong -- check the counterexample against native libc before"
echo "concluding anything."
