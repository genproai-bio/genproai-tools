# METAFlux reference implementation (Huang et al. 2023 Nature Comm)
# Standalone: loads sysdata.rda directly, no package install needed

args <- commandArgs(trailingOnly = TRUE)
input_csv <- args[1]
output_csv <- args[2]

REPO <- "/Users/yuezong.bai/Data/biomedical_tools/metaflux/repo"
library(stringr)
library(stringi)

# Load internal data directly
load(file.path(REPO, "R", "sysdata.rda"))
# Now gene_num, Hgem, iso, multi_comp, simple_comp are in global env

# ── Source the helper functions from calculate_score.R ──
# (but we redefine calculate_reaction_score to avoid METAFlux::: calls)

# Isoenzyme (OR)
calculate_iso_score <- function(x, data, list, gene_num) {
  data_feature <- list[[x]][list[[x]] %in% rownames(data)]
  if (length(data_feature) > 0) {
    vec <- gene_num[data_feature, ]$V1
    expr <- as.matrix(data[data_feature, , drop = FALSE])
    norm <- colSums(sweep(expr, MARGIN = 1, vec, "/"), na.rm = TRUE)
    return(norm)
  } else {
    norm <- matrix(rep(NA, ncol(data)), nrow = 1)
  }
}

# Simple complex (AND)
calculate_simple_comeplex_score <- function(x, data, list, gene_num) {
  data_feature <- list[[x]][list[[x]] %in% rownames(data)]
  if (length(data_feature) > 0) {
    vec <- gene_num[data_feature, ]$V1
    expr <- as.matrix(data[data_feature, , drop = FALSE])
    norm <- apply((1/vec) * expr, 2, min, na.rm = TRUE)
    return(norm)
  } else {
    norm <- matrix(rep(NA, ncol(data)), nrow = 1)
  }
}

# Multi-complex (AND+OR mixed)
calculate_multi_comp <- function(x, data, gene_num) {
  reaction <- multi_comp[[x]]
  com <- stri_detect_fixed(str_extract_all(reaction, "\\([^()]+\\)")[[1]], "or")
  c <- gsub("\\)", "", gsub("\\(", "", str_extract_all(reaction, "\\([^()]+\\)")[[1]]))
  if (unique(com) == TRUE) {
    newiso <- lapply(lapply(c, function(x) unlist(strsplit(x, "or"))), function(x) trimws(x))
    sum_score <- do.call(rbind, lapply(1:length(newiso), calculate_iso_score,
                                       data = data, list = newiso, gene_num = gene_num))
    feature <- trimws(unlist(strsplit(reaction, "and"))[!stri_detect_fixed(unlist(strsplit(reaction, "and")), "or")])
    data_feature <- feature[feature %in% rownames(data)]
    vec <- gene_num[data_feature, ]$V1
    expr <- as.matrix(data[data_feature, , drop = FALSE])
    whole_score <- rbind((1/vec) * expr, sum_score)
    norm <- apply(whole_score, 2, min, na.rm = TRUE)
  } else if (unique(com) == FALSE) {
    newiso <- lapply(lapply(c, function(x) unlist(strsplit(x, "and"))), function(x) trimws(x))
    sum_score <- do.call(rbind, lapply(1:length(newiso), calculate_simple_comeplex_score,
                                       data, list = newiso, gene_num = gene_num))
    feature <- trimws(unlist(strsplit(reaction, "or"))[!stri_detect_fixed(unlist(strsplit(reaction, "or")), "and")])
    data_feature <- feature[feature %in% rownames(data)]
    vec <- gene_num[data_feature, ]$V1
    expr <- as.matrix(data[data_feature, , drop = FALSE])
    upper_score <- (1/vec) * expr
    whole_score <- rbind(upper_score, sum_score)
    norm <- colSums(whole_score, na.rm = TRUE)
  }
}

stdize = function(x, ...) { x / max(x, ...) }

# Main function (standalone, no METAFlux::: calls)
calculate_reaction_score_standalone <- function(data) {
  if (sum(data < 0) > 0) stop("Expression data needs to be all positive")

  features <- rownames(data)
  message(paste0(round(sum(features %in% rownames(gene_num))/3625 * 100, 3),
                 "% metabolic related genes were found......"))

  message("Computing metabolic reaction activity scores......")
  core <- do.call(rbind, lapply(1:length(iso), calculate_iso_score, data = data,
                                list = iso, gene_num = gene_num))
  core2 <- do.call(rbind, lapply(1:length(simple_comp), calculate_simple_comeplex_score,
                                 data, list = simple_comp, gene_num = gene_num))
  core3 <- do.call(rbind, lapply(1:length(multi_comp), calculate_multi_comp,
                                 data = data, gene_num = gene_num))

  message("Preparing for score matrix......")
  rownames(core) <- names(iso)
  rownames(core2) <- names(simple_comp)
  rownames(core3) <- names(multi_comp)

  big_score_matrix <- rbind(core, core2, core3)
  big_score_matrix <- apply(big_score_matrix, 2, stdize, na.rm = TRUE)
  big_score_matrix[is.na(big_score_matrix)] <- 0

  empty_helper <- as.data.frame(Hgem$Reaction)
  colnames(empty_helper) <- "reaction"
  Final_df <- merge(empty_helper, big_score_matrix, all.x = TRUE, by.x = 1, by.y = 0)
  Final_df[is.na(Final_df)] <- 1
  rownames(Final_df) <- Final_df$reaction
  Final_df$reaction <- NULL
  Final_df <- Final_df[Hgem$Reaction, , drop = FALSE]

  if (all.equal(rownames(Final_df), Hgem$Reaction)) {
    message("Metabolic reaction activity scores successfully calculated")
  }

  Final_df[which(Hgem$LB == 0 & Hgem$UB == 0), ] <- 0
  return(Final_df)
}

# ── Run ──
cat("Loading expression matrix...\n")
data <- read.csv(input_csv, row.names = 1, check.names = FALSE)
cat(sprintf("  %d genes × %d samples\n", nrow(data), ncol(data)))

cat("Running METAFlux calculate_reaction_score...\n")
t0 <- proc.time()
mras <- calculate_reaction_score_standalone(data)
elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("  MRAS: %d reactions × %d samples, %.2fs\n", nrow(mras), ncol(mras), elapsed))

write.csv(mras, output_csv)
cat(sprintf("  Saved to %s\n", output_csv))
