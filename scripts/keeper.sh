#!/usr/bin/env bash
# The always-on loop behind forecast.yml. GitHub's cron drops runs under load
# (observed: hours with no firing), so instead of relying on it, one job stays
# alive for ~5.5 h, checks every few minutes what is due, and finally
# dispatches its own successor. Cron only restarts the chain if it breaks.
#
# Each pass: refresh the PowderBench checkout, forecast/submit what's due,
# commit history; once a day make sure the model pipeline ran.
set -uo pipefail
LOOP_MIN="${LOOP_MINUTES:-330}"
SLEEP_S="${SLEEP_SECONDS:-480}"
end=$(( $(date +%s) + LOOP_MIN * 60 ))

commit_history() {
  git add history
  git diff --cached --quiet && return 0
  git commit -qm "history: $(date -u +%Y-%m-%dT%H:%MZ)"
  for i in 1 2 3 4 5; do git pull -q --rebase && git push -q && return 0; sleep $((i * 5)); done
  echo "::warning::could not push history"
}

ran_today() {  # ran_today <workflow file> <earliest UTC hour>
  local today; today=$(date -u +%Y-%m-%d)
  [ "$(date -u +%H)" -lt "$2" ] && return 0  # too early: treat as done
  [ -n "$(gh run list -w "$1" --created ">=${today}" --json databaseId -q '.[0].databaseId' 2>/dev/null)" ]
}

while :; do
  echo "::group::pass $(date -u +%H:%MZ)"
  scripts/checkout_powderbench.sh work/powderbench >/dev/null
  if dpow due | tee due.txt | grep -qv -e "nothing due" -e "^cycle"; then
    python -c "from dpow import store; store.pull('models', store.MODELS, '*.pkl')" 2>/dev/null || echo "no trained models: tier 0"
    dpow run ${DPOW_ARGS:-} || echo "::warning::dpow run reported failures"
    commit_history
  fi
  # GitHub cron drops runs: make sure the daily model pipeline started (after
  # the sibling entries' harvests, which end around 05 UTC)
  ran_today pipeline.yml 6 || { gh workflow run pipeline.yml && echo "dispatched pipeline"; }
  echo "::endgroup::"
  [ $(( $(date +%s) + SLEEP_S )) -ge "$end" ] && break
  sleep "$SLEEP_S"
done

if [ "${CHAIN:-true}" = "true" ]; then
  gh workflow run forecast.yml && echo "dispatched successor"
fi
