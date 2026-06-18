# Prospect League Run Expectancy Matrix

Builds a 24-state run expectancy matrix for the Prospect League 
from PrestoSports XML box scores (2022-2025).


I wrote about about this here: 
https://cornbeltersbaseball.com/run-expectancy-matrix-for-the-prospect-league/

## How it works

- Parses every plate appearance from PrestoSports XML box scores
- Computes runs-from-this-state-on using `innsummary.r - runs_scored_so_far`
- Drops half-innings that didn't end in 3 outs (walk-offs, rain delays)
- Deduplicates games across teams' folders via the venue tag
- Outputs CSVs sliced by year and for Cornbelters games specifically

## Usage

Point the notebook at a folder of PrestoSports XML box scores, run all cells.
Matrices are written to `matrices/`.

## Requirements

- Python 3.9+
- pandas
