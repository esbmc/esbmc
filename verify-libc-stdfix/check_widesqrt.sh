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
# Memory matters here: z3 has been doubling roughly every 6h on the four
# formats with 31-32 fraction bits, while bitwuzla stays flat under 0.5 GB.
# If the z3 total approaches the box size, z3 runs are the preferred OOM
# victims (oom_score_adj=800) so the bitwuzla runs survive.
echo "resident memory per live solver:"
pgrep -f "esbmc oracle_sqrt" 2>/dev/null | while read q; do
  a=$(ps -o args= -p "$q" 2>/dev/null); case "$a" in *oracle_sqrt*) ;; *) continue;; esac
  c=$(ps -o pcpu= -p "$q" 2>/dev/null | tr -d " ")
  [ "${c%%.*}" -lt 50 ] 2>/dev/null && continue
  r=$(ps -o rss= -p "$q" 2>/dev/null | tr -d " ")
  et=$(ps -o etimes= -p "$q" 2>/dev/null | tr -d " ")
  fmt=$(echo "$a" | grep -oE "oracle_sqrt_[a-z]+" | sed "s/oracle_sqrt_//")
  slv=bitwuzla; case "$a" in *--z3*) slv=z3;; esac
  LC_ALL=C printf "  %-5s %-9s %2dh%02dm  %7.2f GB\n" "$fmt" "$slv" \
    $((et/3600)) $(((et%3600)/60)) "$(LC_ALL=C awk -v r=$r "BEGIN{print r/1048576}")"
done | sort -k2,2 -k1,1
free -g | awk "/^Mem/{print \"  box: \"\$3\" GB used / \"\$2\" GB total, \"\$7\" GB available\"}"
echo
echo "Formats: lr=s0.31 ulr=u0.32 k=s16.15 uk=u16.16 lk=s32.31 ulk=u32.32"
echo "A SUCCESSFUL verdict upgrades that format from anchor-validated to"
echo "proved over all inputs. A FAILED verdict would mean the bracket or the"
echo "oracle is wrong -- check the counterexample against native libc before"
echo "concluding anything."
