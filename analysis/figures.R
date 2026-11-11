# Figures for the results site, from results/snapme_v1/. Each figure is written twice to site/public/figures/:
# <name>.svg (light page) and <name>_dark.svg (dark page). Run from the repository root:
#   pixi run -e figures figures
suppressPackageStartupMessages({
  library(jsonlite); library(dplyr); library(tidyr); library(ggplot2); library(scales); library(forcats)
})

res <- "results/snapme_v1"
out <- "site/public/figures"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

themes <- list(
  light = list(bg = "#f2f2f2", ink = "#191919", muted = "#4c4c4c", grid = "#dcdcdc", hl = "#022851", other = "#8c8c8c"),
  dark  = list(bg = "black", ink = "#f2f2f2", muted = "#a8a8a8", grid = "#2a2a2a", hl = "#ffbf00", other = "#8a8a8a")
)

theme_report <- function(p) {
  theme_minimal(base_size = 11, base_family = "sans") +
    theme(plot.background = element_rect(fill = p$bg, colour = NA),
          panel.background = element_rect(fill = p$bg, colour = NA),
          panel.grid.major = element_line(colour = p$grid, linewidth = 0.3),
          panel.grid.minor = element_blank(),
          text = element_text(colour = p$ink), axis.text = element_text(colour = p$muted),
          axis.title = element_text(colour = p$ink, size = 10),
          legend.position = "top", legend.justification = "left", legend.title = element_blank(),
          legend.text = element_text(colour = p$ink), strip.text = element_text(colour = p$ink, hjust = 0),
          plot.margin = margin(8, 12, 8, 8))
}

save_both <- function(make, name, width, height) {
  for (mode in names(themes)) {
    f <- file.path(out, if (mode == "light") paste0(name, ".svg") else paste0(name, "_dark.svg"))
    ggsave(f, make(themes[[mode]]), width = width, height = height, device = svglite::svglite, bg = themes[[mode]]$bg)
  }
}

lb <- fromJSON(file.path(res, "leaderboard.json"), simplifyVector = FALSE)
model_name <- function(r) if (!is.null(r$hf_repo)) sub(".*/", "", r$hf_repo) else r$model
ci <- function(x, i) if (is.null(x)) NA_real_ else as.numeric(x[[i]])
direct <- Filter(function(r) r$pipeline == "direct_v2" && is.null(r$precision_of), lb)

# Figure 1: mean per-photo F1, with intervals from participant resampling, at both grading levels.
# Before graded results are available, show WWEIA category and FNDDS food-code F1.
graded <- Filter(function(r) !is.null(r$same_f1), direct)
if (length(direct) > 0) {
  if (length(graded) > 0) {
    lv <- c("Graded: same food or kind", "Graded: same food")
    d1 <- bind_rows(lapply(graded, function(r) tibble(
      model = model_name(r), level = lv,
      f1 = c(r$type_f1, r$same_f1),
      lo = c(ci(r$type_f1_ci, 1), ci(r$same_f1_ci, 1)),
      hi = c(ci(r$type_f1_ci, 2), ci(r$same_f1_ci, 2)))))
    ord <- d1 |> filter(level == "Graded: same food") |> arrange(f1) |> pull(model)
  } else {
    lv <- c("FNDDS food code", "WWEIA category")
    d1 <- bind_rows(lapply(direct, function(r) tibble(
      model = model_name(r), level = lv,
      f1 = c(r$code_f1, r$wweia_f1),
      lo = c(ci(r$code_f1_ci, 1), ci(r$wweia_f1_ci, 1)),
      hi = c(ci(r$code_f1_ci, 2), ci(r$wweia_f1_ci, 2)))))
    ord <- d1 |> filter(level == "WWEIA category") |> arrange(f1) |> pull(model)
  }
  d1 <- d1 |> mutate(model = factor(model, levels = ord), level = factor(level, levels = lv))
  save_both(function(p) {
    ggplot(d1, aes(x = f1, y = model, colour = level, shape = level)) +
      geom_errorbar(aes(xmin = lo, xmax = hi), orientation = "y", width = 0, linewidth = 0.5, na.rm = TRUE) +
      geom_point(size = 2.2, fill = p$bg, stroke = 0.9) +
      scale_colour_manual(values = c(p$other, p$hl)) + scale_shape_manual(values = c(21, 16)) +
      scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), expand = expansion(mult = c(0, 0.02))) +
      labs(x = "Mean F1 across photos (95% interval)", y = NULL) + theme_report(p)
  }, "fig1_identification", width = 8, height = 1.2 + 0.24 * length(ord))
}

# Figure 2: compare sets of food words using the 2023 protocol, grouped by recorded food-code count.
# Include the five highest-scoring models overall and the two 2023 baselines.
base <- fromJSON(file.path(res, "baselines_2023.json"), simplifyVector = FALSE)
top <- direct[order(-sapply(direct, function(r) if (is.null(r$token_f1_2023)) -1 else r$token_f1_2023))]
top <- head(Filter(function(r) !is.null(r$token_f1_2023_by_food_count), top), 5)
strata <- c("1", "2-3", "4+")
d2 <- bind_rows(
  bind_rows(lapply(top, function(r) tibble(method = model_name(r), kind = "Vision-language model", stratum = strata,
                                            f1 = sapply(strata, function(k) as.numeric(r$token_f1_2023_by_food_count[[k]]))))),
  bind_rows(lapply(base, function(b) tibble(method = paste0(b$method, " (2023)"), kind = "2023 baseline", stratum = strata,
                                             f1 = sapply(strata, function(k) as.numeric(b$by_food_count[[k]]))))))
if (nrow(d2) > 0) {
  # Give each method a row and each food-code group a panel. Close values then have separate labels.
  d2 <- d2 |> mutate(stratum = factor(stratum, levels = strata, labels = c("1 food code recorded", "2-3 food codes recorded", "4 or more food codes")))
  ord2 <- d2 |> group_by(method) |> summarise(m = mean(f1, na.rm = TRUE), .groups = "drop") |> arrange(m) |> pull(method)
  d2 <- d2 |> mutate(method = factor(method, levels = ord2),
                     kind = factor(kind, levels = c("Vision-language model", "2023 baseline")))
  save_both(function(p) {
    ggplot(d2, aes(x = f1, y = method, colour = kind, shape = kind)) +
      geom_point(size = 2.2, fill = p$bg, stroke = 0.9, na.rm = TRUE) +
      facet_wrap(~stratum, nrow = 1) +
      scale_colour_manual(values = c("Vision-language model" = p$hl, "2023 baseline" = p$other)) +
      scale_shape_manual(values = c("Vision-language model" = 16, "2023 baseline" = 21)) +
      scale_x_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1")) +
      labs(x = "Food word agreement (word-set F1, 2023 protocol)", y = NULL) + theme_report(p) +
      theme(panel.spacing.x = unit(1.2, "lines"))
  }, "fig2_food_count", width = 8, height = 1.4 + 0.3 * length(ord2))
}

# Figure 3: estimated versus recorded grams per photo for the model leading Table 1 by same-food F1.
# Before graded results are available, select the model with the highest WWEIA category F1.
if (length(direct) > 0) {
  pool <- if (length(graded) > 0) graded else direct
  key <- if (length(graded) > 0) "same_f1" else "wweia_f1"
  best <- pool[[which.max(sapply(pool, function(r) if (is.null(r[[key]])) -1 else r[[key]]))]]
  det <- fromJSON(file.path(res, "runs", paste0(best$pipeline, "__", best$model, ".json")))
  d3 <- as_tibble(det$photos) |> filter(grams_total >= 10, grams_pred > 0) |>
    mutate(photo_type = recode(photo_type, before = "Before photo", front = "Package front"))
  save_both(function(p) {
    ggplot(d3, aes(x = grams_total, y = grams_pred)) +
      geom_abline(slope = 1, intercept = 0, colour = p$muted, linewidth = 0.4) +
      geom_point(colour = p$hl, alpha = 0.45, size = 1.1, stroke = 0) +
      scale_x_log10(labels = label_number(big.mark = ",")) + scale_y_log10(labels = label_number(big.mark = ",")) +
      facet_wrap(~photo_type) + coord_equal() +
      labs(x = "Grams eaten (ASA24 record)", y = paste0("Grams estimated (", model_name(best), ")")) +
      theme_report(p)
  }, "fig3_amount", width = 8, height = 4.4)
}
cat("figures written to", out, "\n")
