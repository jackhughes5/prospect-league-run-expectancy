# TrackMan Expected Stats Model

# Takes exit velo + launch angle from TrackMan data, trains a random forest
# to predict batted-ball outcomes, then computes expected event rates for
# each batter. Exports a CSV that plugs into the Python lineup simulator.


library(tidyverse)
library(randomForest)
library(caret)

# CONFIG

# Point this at the folder containing your TrackMan CSVs.
TRACKMAN_DIR <- "trackman csv"

# Where to save the output
OUTPUT_FILE  <- "csv output"

# Minimum PAs for a batter to get their own row (others get league-avg)
MIN_PA <- 30

# LOAD & COMBINE ALL CSVs

csv_files <- list.files(TRACKMAN_DIR, pattern = "\\.csv$",
                        recursive = TRUE, full.names = TRUE)
cat(sprintf("Found %d CSV files\n", length(csv_files)))

raw <- csv_files %>%
  map_dfr(~ read_csv(.x, show_col_types = FALSE))

cat(sprintf("Total rows (pitches): %s\n", format(nrow(raw), big.mark = ",")))

# IDENTIFY PLATE APPEARANCES
# Each pitch row belongs to a PA. A PA is "complete" when PitchCall is InPlay,
# or KorBB is Strikeout/Walk, or HitByPitch. 

# Two populations:
#   1: Balls in play -> predicted by exit velo + launch angle
#   2: Strikeouts, walks, HBP -> rates taken directly from the data

# Tag the PA-ending pitch for each plate appearance
pa_enders <- raw %>%
  filter(
    PitchCall == "InPlay" |
      KorBB %in% c("Strikeout", "Walk") |
      PitchCall == "HitByPitch"
  )

cat(sprintf("PA-ending pitches: %s\n", format(nrow(pa_enders), big.mark = ",")))

# BUILD THE BATTED-BALL DATASET
# Filter to balls in play with valid exit velo and launch angle.
# Map PlayResult to the 4 outcome categories that match your simulator.

bip <- pa_enders %>%
  filter(PitchCall == "InPlay",
         !is.na(ExitSpeed),
         !is.na(Angle)) %>%
  mutate(
    outcome = case_when(
      PlayResult == "HomeRun"                        ~ "HR",
      PlayResult == "Triple"                         ~ "3B",
      PlayResult == "Double"                         ~ "2B",
      PlayResult == "Single"                         ~ "1B",
      # Errors: for expected stats, treat as outs
      PlayResult %in% c("Out", "Error",
                        "FieldersChoice", "Sacrifice") ~ "OUT",
      TRUE                                           ~ NA_character_
    )
  ) %>%
  filter(!is.na(outcome))

cat(sprintf("\nBatted balls for model: %s\n", format(nrow(bip), big.mark = ",")))
cat("Outcome distribution:\n")
print(table(bip$outcome))

# TRAIN THE RANDOM FOREST
# RF classifier: Outcome ~ ExitSpeed + Angle

# Make outcome a factor (required for classification RF)
bip$outcome <- factor(bip$outcome, levels = c("OUT", "1B", "2B", "3B", "HR"))

set.seed(42)

# Train/test split for evaluation
train_idx <- createDataPartition(bip$outcome, p = 0.8, list = FALSE)
train_set <- bip[train_idx, ]
test_set  <- bip[-train_idx, ]

cat(sprintf("\nTraining on %d batted balls, testing on %d\n",
            nrow(train_set), nrow(test_set)))

# Train the model
rf_model <- randomForest(
  outcome ~ ExitSpeed + Angle,
  data  = train_set,
  ntree = 500,
  mtry  = 2,
  importance = TRUE
)

print(rf_model)

# EVALUATE
# Confusion matrix on the held-out test set

test_preds <- predict(rf_model, newdata = test_set)
cat("\nTest set confusion matrix:\n")
print(confusionMatrix(test_preds, test_set$outcome))

# Variable importance
cat("\nVariable importance:\n")
print(importance(rf_model))


# GET PREDICTED PROBABILITIES FOR EVERY BATTED BALL
# predict(..., type = "prob") gives P(OUT), P(1B), P(2B), P(3B), P(HR)
# for each batted ball based on its exit velo and launch angle.

bip_probs <- predict(rf_model, newdata = bip, type = "prob") %>%
  as_tibble()

# Attach batter info
bip_with_probs <- bip %>%
  select(Batter, BatterTeam, ExitSpeed, Angle, outcome) %>%
  bind_cols(bip_probs)

# COMPUTE EXPECTED RATES PER BATTER
# For each batter:
#   - K rate and BB rate come from the actual pitch data (not model-based)
#   - For balls in play, average the model's predicted probabilities
#   - Combine: P(event) = P(K)*I(event=OUT) + P(BB)*I(event=BB)
#                        + P(BIP) * avg_model_prob(event)

# PA-level outcomes per batter
batter_pa <- pa_enders %>%
  mutate(
    pa_type = case_when(
      KorBB == "Strikeout"       ~ "K",
      KorBB == "Walk"            ~ "BB",
      PitchCall == "HitByPitch"  ~ "HBP",
      PitchCall == "InPlay"      ~ "BIP",
      TRUE                       ~ NA_character_
    )
  ) %>%
  filter(!is.na(pa_type)) %>%
  group_by(Batter, BatterTeam) %>%
  summarise(
    total_pa = n(),
    n_k      = sum(pa_type == "K"),
    n_bb     = sum(pa_type == "BB") + sum(pa_type == "HBP"),
    n_bip    = sum(pa_type == "BIP"),
    .groups  = "drop"
  ) %>%
  mutate(
    k_rate  = n_k  / total_pa,
    bb_rate = n_bb / total_pa,
    bip_rate = n_bip / total_pa
  )

# Average model-predicted probs per batter (on their BIP only)
batter_xbip <- bip_with_probs %>%
  group_by(Batter, BatterTeam) %>%
  summarise(
    n_bip_modeled = n(),
    x1B = mean(`1B`),
    x2B = mean(`2B`),
    x3B = mean(`3B`),
    xHR = mean(HR),
    xOUT_bip = mean(OUT),
    .groups = "drop"
  )

# Combine into final expected event rates
# These rates sum to 1.0 and map directly to the simulator's event categories:
#   OUT, BB, 1B, 2B, 3B, HR
expected <- batter_pa %>%
  left_join(batter_xbip, by = c("Batter", "BatterTeam")) %>%
  filter(total_pa >= MIN_PA) %>%
  mutate(
    # If a batter has BIP but none were modeled (all Undefined), fall back
    across(c(x1B, x2B, x3B, xHR, xOUT_bip), ~ replace_na(.x, 0)),
    
    # Final expected rates
    xOUT = k_rate + bip_rate * xOUT_bip,
    xBB  = bb_rate,
    x1B_final  = bip_rate * x1B,
    x2B_final  = bip_rate * x2B,
    x3B_final  = bip_rate * x3B,
    xHR_final  = bip_rate * xHR
  ) %>%
  # Normalize to sum to 1.0 (handles small rounding gaps)
  mutate(
    total = xOUT + xBB + x1B_final + x2B_final + x3B_final + xHR_final,
    xOUT      = xOUT / total,
    xBB       = xBB / total,
    x1B_final = x1B_final / total,
    x2B_final = x2B_final / total,
    x3B_final = x3B_final / total,
    xHR_final = xHR_final / total
  ) %>%
  select(
    batter    = Batter,
    team      = BatterTeam,
    pa        = total_pa,
    OUT       = xOUT,
    BB        = xBB,
    `1B`      = x1B_final,
    `2B`      = x2B_final,
    `3B`      = x3B_final,
    HR        = xHR_final
  )

cat(sprintf("\nPlayers with expected rates (>= %d PA): %d\n", MIN_PA, nrow(expected)))

# Preview
cat("\nSample output:\n")
print(expected %>% arrange(desc(pa)) %>% head(10), width = 120)

# EXPORT
write_csv(expected, OUTPUT_FILE)
cat(sprintf("\nWrote %s (%d players)\n", OUTPUT_FILE, nrow(expected)))

# VISUALIZE THE MODEL
# Heatmap of expected bases across EV x LA space

grid <- expand.grid(
  ExitSpeed = seq(40, 115, by = 1),
  Angle     = seq(-30, 50, by = 1)
)

grid_probs <- predict(rf_model, newdata = grid, type = "prob") %>%
  unclass() %>%                # strip the "votes" class
  as.data.frame() %>%          # plain data frame with numeric columns
  bind_cols(grid) %>%
  mutate(xBases = 1 * `1B` + 2 * `2B` + 3 * `3B` + 4 * HR)

p <- ggplot(grid_probs, aes(x = ExitSpeed, y = Angle, fill = xBases)) +
  geom_tile() +
  scale_fill_viridis_c(option = "inferno", name = "Expected\nBases") +
  labs(
    title = "Expected Bases by Exit Velo × Launch Angle",
    subtitle = "Random forest trained on Prospect League TrackMan data",
    x = "Exit Velocity (mph)",
    y = "Launch Angle (°)"
  ) +
  theme_minimal(base_size = 13)

ggsave("png output",
       p, width = 10, height = 6, dpi = 150)
cat("Saved xbases_heatmap.png\n")