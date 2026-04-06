# CytoTRACE reference implementation (Gulati et al. 2020 Science)
# Core algorithm: gene_counts → correlation → top genes → rank-normalized score

args <- commandArgs(trailingOnly = TRUE)
input_csv <- args[1]   # cells x genes expression matrix
output_csv <- args[2]  # output: cell scores

cat("Loading expression matrix...\n")
expr <- as.matrix(read.csv(input_csv, row.names = 1, check.names = FALSE))
cat(sprintf("  %d cells × %d genes\n", nrow(expr), ncol(expr)))

# 1. Gene counts: number of genes with expression > 0 per cell
gene_counts <- rowSums(expr > 0)

# 2. Pearson correlation of each gene with gene_counts
correlations <- apply(expr, 2, function(col) {
  if (sd(col) == 0) return(0)
  cor(col, gene_counts, method = "pearson")
})

# 3. Top 200 positively correlated genes
top_k <- min(200, sum(correlations > 0))
top_idx <- order(correlations, decreasing = TRUE)[1:top_k]
cat(sprintf("  Top %d genes selected\n", top_k))

# 4. Refined score: mean expression of top genes
if (top_k > 0) {
  refined <- rowMeans(expr[, top_idx, drop = FALSE])
} else {
  refined <- gene_counts
}

# 5. Rank normalize to [0, 1]
score <- rank(refined) / length(refined)

# Output
result <- data.frame(
  cell = rownames(expr),
  score = score,
  gene_counts = gene_counts,
  stringsAsFactors = FALSE
)

write.csv(result, output_csv, row.names = FALSE)
cat(sprintf("  Saved %d cell scores to %s\n", nrow(result), output_csv))
