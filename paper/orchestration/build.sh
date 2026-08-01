#!/usr/bin/env bash
# Full LaTeX build for the orchestration paper.
# Runs the complete pdflatex -> bibtex -> pdflatex -> pdflatex cycle so that
# citations, cross-references, and internal hyperlinks all resolve. Use this
# (not a single pdflatex run) before checking links or the bibliography.
#
# Usage:  ./build.sh            # build orchestration_paper
#         ./build.sh clean      # remove aux/log/out/bbl etc. and rebuild from scratch
set -e
cd "$(dirname "$0")"
JOB=orchestration_paper

if [ "$1" = "clean" ]; then
  rm -f "$JOB".aux "$JOB".log "$JOB".out "$JOB".bbl "$JOB".blg \
        "$JOB".toc "$JOB".lof "$JOB".lot "$JOB".lol
fi

# A stale/corrupt .aux is the usual cause of "File ended while scanning
# \@newl@bel" and of links pointing to the wrong page. Guard against it.
pdflatex -interaction=nonstopmode "$JOB".tex || { rm -f "$JOB".aux; pdflatex -interaction=nonstopmode "$JOB".tex; }
bibtex "$JOB" || true
pdflatex -interaction=nonstopmode "$JOB".tex
pdflatex -interaction=nonstopmode "$JOB".tex

echo "---"
if grep -qi "undefined" "$JOB".log; then
  echo "WARNING: undefined references/citations remain:"
  grep -i "undefined" "$JOB".log | head
else
  echo "OK: no undefined references or citations."
fi
grep "Output written" "$JOB".log || true
