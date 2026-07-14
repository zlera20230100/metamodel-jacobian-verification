#!/usr/bin/env Rscript

# Deterministic vector graphical abstract for the SIMPAT submission.
# One R/grid backend produces every preview and export. The design uses only
# square engineering boxes, orthogonal connectors, traceable values, and the
# manuscript-wide blue/grey/warm palette.

suppressPackageStartupMessages({
  library(grid)
  library(svglite)
  library(ragg)
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
here <- if (length(file_arg)) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), winslash = "/"))
} else {
  normalizePath(getwd(), winslash = "/")
}

operating <- read.csv(file.path(here, "graphical_abstract_operating_points.csv"),
                      check.names = FALSE, stringsAsFactors = FALSE)
signs_df <- read.csv(file.path(here, "graphical_abstract_sign_matrix.csv"),
                     check.names = FALSE)
signs <- as.matrix(signs_df[, -1])

W_MM <- 183
H_MM <- 73
W_IN <- W_MM / 25.4
H_IN <- H_MM / 25.4
FONT <- "Times New Roman"

COL <- list(
  blue = "#3775BA",
  blue_mid = "#6B9AC4",
  blue_light = "#DCE8F1",
  orange = "#C76B3C",
  orange_light = "#F5E7DF",
  grey_fill = "#F2F2F2",
  grey_line = "#B8B8B8",
  grey_text = "#666666",
  dark = "#4D4D4D",
  ink = "#1B1B1B",
  white = "#FFFFFF"
)

boxes <- new.env(parent = emptyenv())

txt <- function(x, y, label, size = 6, face = "plain", col = COL$ink,
                just = c("left", "centre"), rot = 0, lineheight = 0.95) {
  grid.text(label, x = unit(x, "npc"), y = unit(y, "npc"),
            just = just, rot = rot,
            gp = gpar(fontfamily = FONT, fontsize = size, fontface = face,
                      col = col, lineheight = lineheight))
}

rule <- function(x0, y0, x1, y1, col = COL$grey_line, lwd = 0.7) {
  grid.lines(unit(c(x0, x1), "npc"), unit(c(y0, y1), "npc"),
             gp = gpar(col = col, lwd = lwd, lineend = "butt"))
}

rect_node <- function(id, x0, y0, x1, y1, fill = COL$white,
                      stroke = COL$dark, lwd = 0.8) {
  grid.rect(x = unit((x0 + x1) / 2, "npc"), y = unit((y0 + y1) / 2, "npc"),
            width = unit(x1 - x0, "npc"), height = unit(y1 - y0, "npc"),
            gp = gpar(fill = fill, col = stroke, lwd = lwd, linejoin = "mitre"))
  assign(id, c(x0 = x0, y0 = y0, x1 = x1, y1 = y1), envir = boxes)
}

on_boundary <- function(p, b, tol = 1e-8) {
  on_x <- (abs(p[1] - b["x0"]) < tol || abs(p[1] - b["x1"]) < tol) &&
    p[2] >= b["y0"] - tol && p[2] <= b["y1"] + tol
  on_y <- (abs(p[2] - b["y0"]) < tol || abs(p[2] - b["y1"]) < tol) &&
    p[1] >= b["x0"] - tol && p[1] <= b["x1"] + tol
  on_x || on_y
}

inside_strict <- function(x, y, b, tol = 1e-6) {
  x > b["x0"] + tol && x < b["x1"] - tol &&
    y > b["y0"] + tol && y < b["y1"] - tol
}

validate_route <- function(points, start_id = NULL, end_id = NULL) {
  stopifnot(ncol(points) == 2, nrow(points) >= 2)
  d <- diff(points)
  if (any(abs(d[, 1]) > 1e-10 & abs(d[, 2]) > 1e-10))
    stop("Connector contains a non-orthogonal segment")
  if (!is.null(start_id) && !on_boundary(points[1, ], get(start_id, boxes)))
    stop("Connector does not start on the source-node boundary: ", start_id)
  if (!is.null(end_id) && !on_boundary(points[nrow(points), ], get(end_id, boxes)))
    stop("Connector does not end on the target-node boundary: ", end_id)

  ids <- ls(boxes)
  for (i in seq_len(nrow(points) - 1)) {
    t <- seq(0.02, 0.98, length.out = 60)
    xs <- points[i, 1] + t * (points[i + 1, 1] - points[i, 1])
    ys <- points[i, 2] + t * (points[i + 1, 2] - points[i, 2])
    for (id in ids) {
      if (id %in% c(start_id, end_id)) next
      b <- get(id, boxes)
      if (any(mapply(inside_strict, xs, ys, MoreArgs = list(b = b))))
        stop("Connector intersects node interior: ", id)
    }
  }
  invisible(TRUE)
}

ortho_arrow <- function(points, start_id = NULL, end_id = NULL,
                        col = COL$dark, lwd = 0.9, head_mm = 1.8) {
  points <- as.matrix(points)
  validate_route(points, start_id, end_id)
  grid.lines(unit(points[, 1], "npc"), unit(points[, 2], "npc"),
             arrow = arrow(type = "closed", ends = "last",
                           length = unit(head_mm, "mm")),
             gp = gpar(col = col, fill = col, lwd = lwd,
                       lineend = "butt", linejoin = "mitre"))
}

stage_heading <- function(x, y, number, label) {
  grid.rect(unit(x, "npc"), unit(y, "npc"),
            width = unit(0.022, "npc"), height = unit(0.030, "npc"),
            gp = gpar(fill = COL$blue, col = COL$blue, lwd = 0.6))
  txt(x, y, number, size = 6.2, face = "bold", col = COL$white,
      just = c("centre", "centre"))
  txt(x + 0.018, y, label, size = 7.2, face = "bold")
}

draw_figure <- function() {
  grid.newpage()
  pushViewport(viewport(xscale = c(0, 1), yscale = c(0, 1),
                        gp = gpar(fontfamily = FONT)))

  # Header
  txt(0.026, 0.950, "Reference-calibrated verification of metamodel design Jacobians",
      size = 12.2, face = "bold", just = c("left", "top"))
  txt(0.026, 0.898,
      "Rank components, verify only where calibrated evidence is insufficient, and report empirical risk at the actual simulation cost.",
      size = 6.6, col = COL$grey_text, just = c("left", "top"))
  rule(0.026, 0.855, 0.974, 0.855, COL$ink, 0.8)

  # Stage separators establish hierarchy without rounded card layouts.
  rule(0.708, 0.160, 0.708, 0.820, COL$grey_line, 0.65)
  stage_heading(0.040, 0.812, "1", "Cross-retrain evidence")
  stage_heading(0.335, 0.812, "2", "Calibrate and allocate")
  stage_heading(0.733, 0.812, "3", "Report the operating point")

  # Stage 1: deterministic sign matrix from traceable CSV input.
  txt(0.050, 0.755, "Autodiff signs from M independent retrains", size = 5.8)
  x0 <- 0.075; y_top <- 0.700; cw <- 0.029; ch <- 0.040
  gx <- 0.006; gy <- 0.010
  for (r in seq_len(nrow(signs))) {
    yy <- y_top - (r - 1) * (ch + gy)
    txt(0.061, yy, as.character(signs_df$retrain[r]), size = 4.7,
        col = COL$grey_text, just = c("right", "centre"))
    for (k in seq_len(ncol(signs))) {
      xx <- x0 + (k - 1) * (cw + gx)
      positive <- signs[r, k] > 0
      grid.rect(unit(xx, "npc"), unit(yy, "npc"),
                width = unit(cw, "npc"), height = unit(ch, "npc"),
                gp = gpar(fill = if (positive) COL$blue_light else COL$orange_light,
                          col = if (positive) COL$blue else COL$orange,
                          lwd = 0.65))
      txt(xx, yy, if (positive) "+" else "−", size = 6.0, face = "bold",
          col = if (positive) COL$blue else COL$orange,
          just = c("centre", "centre"))
    }
  }
  for (k in seq_len(ncol(signs))) {
    xx <- x0 + (k - 1) * (cw + gx)
    txt(xx, 0.438, paste0("g", k), size = 4.6, col = COL$grey_text,
        just = c("centre", "centre"))
  }
  txt(0.052, 0.570, "retrain", size = 4.7, col = COL$grey_text,
      rot = 90, just = c("centre", "centre"))

  rect_node("scores", 0.050, 0.235, 0.285, 0.380,
            fill = COL$white, stroke = COL$blue, lwd = 0.9)
  txt(0.062, 0.352, "Candidate component scores", size = 5.9, face = "bold")
  txt(0.062, 0.316, "sign agreement  aₖ", size = 5.4, col = COL$blue)
  txt(0.062, 0.282, "signal-to-noise  uₖ", size = 5.4)
  txt(0.205, 0.282, "magnitude  mₖ", size = 5.4)

  # Stage 2: two calibration inputs, one prespecified operating rule.
  rect_node("reference", 0.355, 0.685, 0.662, 0.758,
            fill = COL$grey_fill, stroke = COL$dark, lwd = 0.8)
  txt(0.508, 0.729, "Independent reference-labelled calibration queries",
      size = 5.8, face = "bold", just = c("centre", "centre"))
  txt(0.508, 0.701, "numerically screened central differences",
      size = 4.9, col = COL$grey_text, just = c("centre", "centre"))

  rect_node("loss", 0.388, 0.565, 0.630, 0.637,
            fill = COL$white, stroke = COL$dark, lwd = 0.8)
  txt(0.509, 0.608, "Select the score for the intended loss",
      size = 5.7, face = "bold", just = c("centre", "centre"))
  txt(0.509, 0.581, "ranking AUC  |  risk–coverage  |  budgeted gradient error",
      size = 4.7, col = COL$grey_text, just = c("centre", "centre"))

  rect_node("rule", 0.375, 0.440, 0.643, 0.515,
            fill = COL$blue_light, stroke = COL$blue, lwd = 0.9)
  txt(0.509, 0.484, "Calibrated threshold τₛ or fixed budget B",
      size = 5.8, face = "bold", col = COL$blue,
      just = c("centre", "centre"))
  txt(0.509, 0.456, "verify the B lowest-ranked components",
      size = 4.9, col = COL$dark, just = c("centre", "centre"))

  rect_node("accept", 0.348, 0.280, 0.465, 0.355,
            fill = COL$white, stroke = COL$blue, lwd = 1.0)
  txt(0.4065, 0.327, "ACCEPT", size = 6.0, face = "bold", col = COL$blue,
      just = c("centre", "centre"))
  txt(0.4065, 0.299, "retain mean Jₖ", size = 4.8,
      just = c("centre", "centre"))

  rect_node("verify", 0.548, 0.280, 0.665, 0.355,
            fill = COL$white, stroke = COL$orange, lwd = 1.0)
  txt(0.6065, 0.327, "VERIFY", size = 6.0, face = "bold", col = COL$orange,
      just = c("centre", "centre"))
  txt(0.6065, 0.299, "central reference FD", size = 4.8,
      just = c("centre", "centre"))

  rect_node("hybrid", 0.360, 0.170, 0.540, 0.235,
            fill = COL$white, stroke = COL$dark, lwd = 0.8)
  txt(0.450, 0.221, "hybrid design Jacobian", size = 5.2, face = "bold",
      just = c("centre", "centre"))
  vec_cols <- c(COL$blue, COL$orange, COL$orange, COL$blue, COL$blue, COL$orange)
  for (i in seq_along(vec_cols)) {
    xx <- 0.382 + (i - 1) * 0.027
    grid.rect(unit(xx, "npc"), unit(0.190, "npc"),
              width = unit(0.020, "npc"), height = unit(0.023, "npc"),
              gp = gpar(fill = COL$white, col = vec_cols[i], lwd = 0.85))
  }

  rect_node("abstain", 0.563, 0.170, 0.675, 0.235,
            fill = COL$grey_fill, stroke = COL$grey_text, lwd = 0.8)
  txt(0.619, 0.211, "ABSTAIN / REFINE", size = 5.0, face = "bold",
      col = COL$grey_text, just = c("centre", "centre"))
  txt(0.619, 0.187, "if the screen fails", size = 4.5,
      col = COL$grey_text, just = c("centre", "centre"))

  # Orthogonal routes meet node boundaries and remain clear of labels.
  ortho_arrow(rbind(c(0.285, 0.308), c(0.335, 0.308), c(0.335, 0.601), c(0.388, 0.601)),
              "scores", "loss", COL$blue, 0.85)
  ortho_arrow(rbind(c(0.508, 0.685), c(0.508, 0.637)),
              "reference", "loss", COL$dark, 0.8)
  ortho_arrow(rbind(c(0.509, 0.565), c(0.509, 0.515)),
              "loss", "rule", COL$dark, 0.8)
  ortho_arrow(rbind(c(0.445, 0.440), c(0.445, 0.397), c(0.4065, 0.397), c(0.4065, 0.355)),
              "rule", "accept", COL$blue, 0.9)
  ortho_arrow(rbind(c(0.575, 0.440), c(0.575, 0.397), c(0.6065, 0.397), c(0.6065, 0.355)),
              "rule", "verify", COL$orange, 0.9)
  txt(0.4065, 0.408, "reliable", size = 4.6, col = COL$blue,
      just = c("centre", "centre"))
  txt(0.6065, 0.408, "least reliable", size = 4.6, col = COL$orange,
      just = c("centre", "centre"))
  ortho_arrow(rbind(c(0.4065, 0.280), c(0.4065, 0.235)),
              "accept", "hybrid", COL$blue, 0.9)
  ortho_arrow(rbind(c(0.6065, 0.280), c(0.6065, 0.255), c(0.500, 0.255), c(0.500, 0.235)),
              "verify", "hybrid", COL$orange, 0.9)
  ortho_arrow(rbind(c(0.665, 0.318), c(0.689, 0.318), c(0.689, 0.2025), c(0.675, 0.2025)),
              "verify", "abstain", COL$grey_text, 0.75)

  # Stage 3: measured external operating points, loaded from CSV.
  txt(0.744, 0.757, "External regimes at τ = 0.9",
      size = 5.8, face = "bold")
  txt(0.744, 0.710, "benchmark", size = 4.7, face = "bold", col = COL$grey_text)
  txt(0.873, 0.710, "coverage", size = 4.6, face = "bold", col = COL$grey_text,
      just = c("centre", "centre"))
  txt(0.917, 0.710, "risk", size = 4.6, face = "bold", col = COL$grey_text,
      just = c("centre", "centre"))
  txt(0.970, 0.710, "evals", size = 4.5, face = "bold", col = COL$grey_text,
      just = c("right", "centre"))
  rule(0.744, 0.687, 0.970, 0.687, COL$ink, 0.6)

  row_y <- c(0.641, 0.580, 0.519)
  for (i in seq_len(nrow(operating))) {
    txt(0.744, row_y[i], operating$benchmark[i], size = 4.9)
    txt(0.873, row_y[i], sprintf("%.1f%%", operating$coverage_pct[i]),
        size = 5.0, face = "bold", col = COL$blue,
        just = c("centre", "centre"))
    txt(0.917, row_y[i], sprintf("%.1f%%", operating$false_trust_risk_pct[i]),
        size = 5.0, face = "bold", col = COL$orange,
        just = c("centre", "centre"))
    txt(0.970, row_y[i], sprintf("%.2f/%d", operating$selective_evals[i],
                                 operating$complete_fd_evals[i]),
        size = 4.8, just = c("right", "centre"))
    rule(0.744, row_y[i] - 0.031, 0.970, row_y[i] - 0.031,
         COL$grey_fill, 1.0)
  }
  txt(0.970, 0.470, "evals: selective / complete central FD", size = 4.2,
      col = COL$grey_text, just = c("right", "centre"))

  rect_node("loss_report", 0.744, 0.294, 0.970, 0.435,
            fill = COL$white, stroke = COL$grey_line, lwd = 0.7)
  txt(0.756, 0.408, "Score choice follows the deployment loss", size = 5.5, face = "bold")
  txt(0.756, 0.368, "sign-correctness AUC", size = 4.8, col = COL$grey_text)
  txt(0.960, 0.368, "SNR / magnitude", size = 4.9, col = COL$blue,
      face = "bold", just = c("right", "centre"))
  txt(0.756, 0.329, "fixed-budget gradient error", size = 4.8, col = COL$grey_text)
  txt(0.960, 0.329, "sign ordering", size = 4.9, col = COL$orange,
      face = "bold", just = c("right", "centre"))

  rect_node("evidence", 0.744, 0.170, 0.970, 0.252,
            fill = COL$grey_fill, stroke = COL$grey_text, lwd = 0.7)
  txt(0.857, 0.222, "3,580 / 3,580 external labels passed numerical screens",
      size = 4.9, face = "bold", just = c("centre", "centre"))
  txt(0.857, 0.190, "unresolved antenna derivatives were excluded",
      size = 4.6, col = COL$grey_text, just = c("centre", "centre"))

  # Bottom structural boundary: evidence limitation, not decoration.
  rule(0.026, 0.125, 0.974, 0.125, COL$orange, 1.0)
  txt(0.026, 0.082, "Structural boundary", size = 5.7, face = "bold", col = COL$orange)
  txt(0.140, 0.082,
      "positive rescaling leaves sign agreement and SNR unchanged; shared bias can make every retrain agree and still be wrong",
      size = 5.3)
  txt(0.974, 0.042, "calibrated triage  ≠  correctness certificate",
      size = 5.2, face = "bold", col = COL$grey_text,
      just = c("right", "centre"))

  popViewport()
}

base <- file.path(here, "graphical_abstract_simpat")

svglite::svglite(paste0(base, ".svg"), width = W_IN, height = H_IN,
                 bg = "white", system_fonts = list(sans = FONT, serif = FONT))
draw_figure()
dev.off()

grDevices::cairo_pdf(paste0(base, ".pdf"), width = W_IN, height = H_IN,
                     family = FONT, bg = "white")
draw_figure()
dev.off()

ragg::agg_png(paste0(base, ".png"), width = W_IN, height = H_IN,
              units = "in", res = 300, background = "white", scaling = 1)
draw_figure()
dev.off()

ragg::agg_tiff(paste0(base, ".tiff"), width = W_IN, height = H_IN,
               units = "in", res = 600, background = "white",
               compression = "lzw", scaling = 1)
draw_figure()
dev.off()

cat(sprintf("wrote %s.{svg,pdf,png,tiff}\n", base))
