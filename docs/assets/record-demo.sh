#!/usr/bin/env bash
# Driver for the README demo GIF. Regenerate from the repo root with:
#   asciinema rec --overwrite -c "bash docs/assets/record-demo.sh" /tmp/demo.cast
#   agg --font-size 16 --speed 1 /tmp/demo.cast docs/assets/demo.gif
# Requires: brew install asciinema agg; package installed in .venv

set -e
export PATH="$PWD/.venv/bin:$PATH"
D=$(mktemp -d)

type_cmd() {
  printf '\033[1;32m❯\033[0m '
  local cmd="$1"
  for ((i = 0; i < ${#cmd}; i++)); do
    printf '%s' "${cmd:$i:1}"
    sleep 0.018
  done
  printf '\n'
  sleep 0.3
}

say() {
  printf '\033[2m%s\033[0m\n' "$1"
  sleep 0.9
}

run() {
  type_cmd "$1"
  eval "$1"
  sleep "$2"
}

say "# Agent Memory — a decision layer, not just a retriever"
run "agent-memory --data-dir \$D remember 'How do I reset my password?' 'Go to Settings → Security → Reset Password.'" 1.2
run "agent-memory --data-dir \$D remember 'What payment methods do you support?' 'We accept Visa, Mastercard, and PayPal.'" 1.2
echo
say "# Exact repeat → REPLAY the stored answer"
run "agent-memory --data-dir \$D resolve 'How do I reset my password?'" 2.2
echo
say "# Shares the word 'support' — a naive retriever replays the PayPal answer…"
run "agent-memory --data-dir \$D resolve 'Does the platform support two-factor authentication?' --explain" 4.5
echo
say "# action: none — and it shows you exactly why."
sleep 2
