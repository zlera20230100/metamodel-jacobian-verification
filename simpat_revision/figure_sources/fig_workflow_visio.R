# Publication figure source for Fig. 1.
# Backend: base R grid only. The layout is a deterministic Visio-style
# engineering diagram with square nodes and orthogonal connectors.

suppressPackageStartupMessages(library(grid))

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[1]) else "fig_workflow_visio.R"
here <- dirname(normalizePath(script_path, winslash = "/", mustWork = FALSE))

nodes <- read.csv(file.path(here, "fig_workflow_nodes.csv"),
                  stringsAsFactors = FALSE, check.names = FALSE)
edges <- read.csv(file.path(here, "fig_workflow_edges.csv"),
                  stringsAsFactors = FALSE, check.names = FALSE)

W_MM <- 183
H_MM <- 84
W_IN <- W_MM / 25.4
H_IN <- H_MM / 25.4

COL <- c(
  ink = "#222222",
  blue = "#3775BA",
  blue_mid = "#6B9AC4",
  blue_light = "#A9C5DF",
  blue_pale = "#DCE8F1",
  warm = "#C76B3C",
  warm_pale = "#F5E7DF",
  grey = "#4D4D4D",
  grey_light = "#CFCECE",
  grey_pale = "#F4F4F4",
  phase = "#FAFBFC",
  white = "#FFFFFF"
)

fill_value <- function(key) unname(COL[[key]])

phase_box <- function(x0, x1, label, header_fill) {
  grid.rect(x = unit((x0 + x1) / 2, "npc"), y = unit(0.50, "npc"),
            width = unit(x1 - x0, "npc"), height = unit(0.92, "npc"),
            gp = gpar(fill = COL[["phase"]], col = COL[["grey_light"]], lwd = 0.8))
  grid.rect(x = unit((x0 + x1) / 2, "npc"), y = unit(0.925, "npc"),
            width = unit(x1 - x0, "npc"), height = unit(0.07, "npc"),
            gp = gpar(fill = header_fill, col = COL[["grey_light"]], lwd = 0.8))
  grid.text(label, x = unit(x0 + 0.012, "npc"), y = unit(0.925, "npc"),
            just = "left", gp = gpar(fontfamily = "Times New Roman",
                                      fontsize = 8.4, fontface = "bold",
                                      col = COL[["ink"]]))
}

route <- function(path, colour = COL[["grey"]], lwd = 0.9, label = "",
                  label_x = NA_real_, label_y = NA_real_) {
  pts <- strsplit(path, "[|]", perl = TRUE)[[1]]
  xy <- do.call(rbind, lapply(pts, function(p) as.numeric(strsplit(p, ";", fixed = TRUE)[[1]])))
  grid.lines(x = unit(xy[, 1], "npc"), y = unit(xy[, 2], "npc"),
             gp = gpar(col = colour, lwd = lwd, lineend = "butt", linejoin = "mitre"),
             arrow = arrow(type = "closed", angle = 22, length = unit(1.5, "mm")))
  if (nzchar(label) && is.finite(label_x) && is.finite(label_y)) {
    # Labels are offset into clear space.  No white mask is used because a
    # label must never interrupt or conceal a connector.
    grid.text(label, x = unit(label_x, "npc"), y = unit(label_y, "npc"),
              gp = gpar(fontfamily = "Times New Roman", fontsize = 6.3,
                        col = colour))
  }
}

segment_hits_rect <- function(x1, y1, x2, y2, row, eps = 1e-6) {
  left <- as.numeric(row[["x"]]) - as.numeric(row[["w"]]) / 2
  right <- as.numeric(row[["x"]]) + as.numeric(row[["w"]]) / 2
  bottom <- as.numeric(row[["y"]]) - as.numeric(row[["h"]]) / 2
  top <- as.numeric(row[["y"]]) + as.numeric(row[["h"]]) / 2
  if (abs(x1 - x2) < eps) {
    inside_axis <- x1 > left + eps && x1 < right - eps
    overlap <- max(min(y1, y2), bottom + eps) < min(max(y1, y2), top - eps)
  } else if (abs(y1 - y2) < eps) {
    inside_axis <- y1 > bottom + eps && y1 < top - eps
    overlap <- max(min(x1, x2), left + eps) < min(max(x1, x2), right - eps)
  } else {
    stop("Connector segments must be orthogonal.")
  }
  inside_axis && overlap
}

validate_geometry <- function() {
  for (i in seq_len(nrow(edges))) {
    e <- edges[i, ]
    pts <- strsplit(e[["path"]], "[|]", perl = TRUE)[[1]]
    xy <- do.call(rbind, lapply(pts, function(p) {
      as.numeric(strsplit(p, ";", fixed = TRUE)[[1]])
    }))
    for (j in seq_len(nrow(xy) - 1L)) {
      for (k in seq_len(nrow(nodes))) {
        if (segment_hits_rect(xy[j, 1], xy[j, 2], xy[j + 1, 1], xy[j + 1, 2],
                              nodes[k, ])) {
          stop(sprintf("Geometry QA failed: connector %s crosses node %s.",
                       e[["id"]], nodes[k, "id"]))
        }
      }
    }
  }
  invisible(TRUE)
}

math_detail <- function(id, x, y) {
  expr <- switch(id,
    A2 = expression(hat(J)[k]^(m) == partialdiff*hat(F)^(m)/partialdiff*g[k]),
    C1 = expression(s[k]~"at one deployment query"),
    C2 = expression(s[k] >= tau[s]*" ?"),
    C3 = expression("use"~bar(J)[k]),
    C5 = expression("hybrid"~tilde(J)~"; risk; coverage; calls"),
    NULL
  )
  if (!is.null(expr)) {
    grid.text(expr, x = unit(x, "npc"), y = unit(y, "npc"),
              gp = gpar(fontfamily = "Times New Roman", fontsize = 6.45,
                        col = COL[["grey"]]))
  }
}

draw_node <- function(row) {
  x <- as.numeric(row[["x"]]); y <- as.numeric(row[["y"]])
  w <- as.numeric(row[["w"]]); h <- as.numeric(row[["h"]])
  edge <- fill_value(row[["edge"]]); fill <- fill_value(row[["fill"]])
  is_decision <- identical(row[["shape"]], "diamond")

  if (is_decision) {
    grid.polygon(x = unit(c(x, x + w / 2, x, x - w / 2), "npc"),
                 y = unit(c(y + h / 2, y, y - h / 2, y), "npc"),
                 gp = gpar(fill = fill, col = edge, lwd = 1.0, linejoin = "mitre"))
  } else {
    grid.rect(x = unit(x, "npc"), y = unit(y, "npc"),
              width = unit(w, "npc"), height = unit(h, "npc"),
              gp = gpar(fill = fill, col = edge, lwd = 0.9, linejoin = "mitre"))
  }

  title_y <- y + if (is_decision) 0.013 else h * 0.20
  detail_y <- y - if (is_decision) 0.015 else h * 0.18
  title_col <- if (row[["edge"]] == "warm") COL[["warm"]] else COL[["ink"]]
  grid.text(row[["title"]], x = unit(x, "npc"), y = unit(title_y, "npc"),
            gp = gpar(fontfamily = "Times New Roman", fontsize = 7.25,
                      fontface = "bold", col = title_col),
            just = "centre")

  if (as.integer(row[["math"]]) == 1L) {
    math_detail(row[["id"]], x, detail_y)
  } else if (nzchar(row[["detail"]])) {
    detail <- gsub("\\\\n", "\n", row[["detail"]], fixed = TRUE)
    grid.text(detail, x = unit(x, "npc"), y = unit(detail_y, "npc"),
              gp = gpar(fontfamily = "Times New Roman", fontsize = 6.25,
                        col = COL[["grey"]], lineheight = 0.92),
              just = "centre")
  }
}

draw_figure <- function() {
  grid.newpage()
  grid.rect(gp = gpar(fill = "white", col = NA))

  phase_box(0.015, 0.245, "1  ENSEMBLE", COL[["blue_pale"]])
  phase_box(0.265, 0.610, "2  REFERENCE CALIBRATION", COL[["grey_pale"]])
  phase_box(0.630, 0.985, "3  DEPLOYMENT", COL[["blue_pale"]])

  # Connectors are drawn before nodes so no line can cover node text.
  for (i in seq_len(nrow(edges))) {
    e <- edges[i, ]
    route(e[["path"]], fill_value(e[["colour"]]), as.numeric(e[["lwd"]]),
          e[["label"]], as.numeric(e[["label_x"]]), as.numeric(e[["label_y"]]))
  }

  for (i in seq_len(nrow(nodes))) draw_node(nodes[i, ])

  grid.text("Reference calls", x = unit(0.278, "npc"), y = unit(0.842, "npc"),
            just = "left", gp = gpar(fontfamily = "Times New Roman",
                                      fontsize = 6.0, fontface = "italic",
                                      col = COL[["grey"]]))
  grid.text("0 reference calls to compute the selected score",
            x = unit(0.646, "npc"), y = unit(0.842, "npc"), just = "left",
            gp = gpar(fontfamily = "Times New Roman", fontsize = 6.0,
                      fontface = "italic", col = COL[["grey"]]))
}

export_one <- function(ext) {
  out <- file.path(here, paste0("fig_workflow.", ext))
  if (ext == "pdf") {
    cairo_pdf(out, width = W_IN, height = H_IN, family = "Times New Roman",
              bg = "white", onefile = TRUE)
  } else if (ext == "svg") {
    svg(out, width = W_IN, height = H_IN, family = "Times New Roman",
        bg = "white", onefile = TRUE)
  } else if (ext == "png") {
    png(out, width = W_IN, height = H_IN, units = "in", res = 300,
        type = "cairo-png", bg = "white")
  } else if (ext == "tiff") {
    tiff(out, width = W_IN, height = H_IN, units = "in", res = 600,
         compression = "lzw", type = "cairo", bg = "white")
  }
  draw_figure()
  dev.off()
  message("wrote ", out)
}

validate_geometry()
invisible(lapply(c("pdf", "svg", "png", "tiff"), export_one))
