#!/usr/bin/env sh
# The demo from any folder:
#   ./demo.sh                          a tour of one N, S, V and F beat (Enter between beats)
#   ./demo.sh S                        one example beat: N, S, V or F
#   ./demo.sh --list                   beat positions to choose from
#   ./demo.sh --record 118 --beat 112  any validation beat
# Gemini writes the review when GEMINI_API_KEY is set; without it the same review is filled in offline.
unset VIRTUAL_ENV  # another active environment would only make uv print a warning
cd "$(dirname "$0")" && exec uv run python -m ecgaug.demo "$@"
