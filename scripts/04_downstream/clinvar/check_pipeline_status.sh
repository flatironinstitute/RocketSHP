#!/bin/bash
# Quick status checker for ClinVar analysis pipeline

echo "=========================================="
echo "ClinVar Pipeline Status"
echo "=========================================="
echo ""

# Check for output files
echo "📁 Output Files Status:"
echo ""

OUTPUT_DIR="$HOME/Projects/rocketshp/data/processed/clinvar"
REPORTS_DIR="$HOME/Projects/rocketshp/reports/figures/clinvar"

check_file() {
    if [ -f "$1" ]; then
        SIZE=$(du -h "$1" | cut -f1)
        echo "  ✓ $2 ($SIZE)"
        return 0
    else
        echo "  ✗ $2 (missing)"
        return 1
    fi
}

# Step 1
check_file "$OUTPUT_DIR/variant_dataset.csv" "Step 1: Variant Dataset"

# Step 2
check_file "$OUTPUT_DIR/wildtype_predictions.h5" "Step 2: WT Predictions"
if [ -f "$OUTPUT_DIR/wt_predictions_checkpoint.txt" ]; then
    WT_COUNT=$(wc -l < "$OUTPUT_DIR/wt_predictions_checkpoint.txt")
    echo "    └─ Checkpoint: $WT_COUNT proteins processed"
fi

# Step 3
check_file "$OUTPUT_DIR/variant_predictions.h5" "Step 3: Variant Predictions"
if [ -f "$OUTPUT_DIR/variant_predictions_checkpoint.txt" ]; then
    VAR_COUNT=$(wc -l < "$OUTPUT_DIR/variant_predictions_checkpoint.txt")
    echo "    └─ Checkpoint: $VAR_COUNT variants processed"
fi

# Step 4
check_file "$OUTPUT_DIR/variant_features.csv" "Step 4: Variant Features"

# Step 5
check_file "$OUTPUT_DIR/statistical_results.json" "Step 5: Statistical Results"

# Step 6
echo ""
echo "  Step 6: Visualizations"
check_file "$REPORTS_DIR/distributions.png" "    - Distributions"
check_file "$REPORTS_DIR/roc_curves.png" "    - ROC Curves"
check_file "$REPORTS_DIR/feature_importance.png" "    - Feature Importance"
check_file "$REPORTS_DIR/summary_table.png" "    - Summary Table"

if [ -d "$REPORTS_DIR/network_examples" ]; then
    NET_COUNT=$(ls -1 "$REPORTS_DIR/network_examples"/*.png 2>/dev/null | wc -l)
    if [ $NET_COUNT -gt 0 ]; then
        echo "  ✓   - Network Examples ($NET_COUNT files)"
    fi
fi

# Step 7
echo ""
check_file "$REPORTS_DIR/clinvar_pathogenicity_report.html" "Step 7: Final Report"

echo ""
echo "=========================================="
echo "🔄 Running Jobs:"
echo "=========================================="
echo ""

# Check for running jobs
JOBS=$(squeue -u $USER -n "clinvar_*" --format="%.18i %.15j %.8T %.10M %.9l %.6D %R" 2>/dev/null)

if [ -z "$JOBS" ]; then
    echo "  No ClinVar jobs currently running"
else
    echo "$JOBS"
fi

echo ""
echo "=========================================="
echo "📊 Recent Logs:"
echo "=========================================="
echo ""

# Show most recent log files
LOGS_DIR="$HOME/Projects/rocketshp/slurm_logs"
if [ -d "$LOGS_DIR" ]; then
    RECENT_LOGS=$(ls -t "$LOGS_DIR"/clinvar_*.out 2>/dev/null | head -3)
    if [ -n "$RECENT_LOGS" ]; then
        for LOG in $RECENT_LOGS; do
            BASENAME=$(basename "$LOG")
            LAST_LINE=$(tail -1 "$LOG" 2>/dev/null)
            echo "  $BASENAME"
            echo "    └─ $LAST_LINE"
        done
    else
        echo "  No log files found"
    fi
fi

echo ""
echo "=========================================="

# Quick summary
echo ""
if [ -f "$REPORTS_DIR/clinvar_pathogenicity_report.html" ]; then
    echo "✅ Pipeline complete! View report at:"
    echo "   file://$REPORTS_DIR/clinvar_pathogenicity_report.html"
else
    echo "⏳ Pipeline in progress. Use 'squeue -u $USER' to monitor jobs."
fi
echo ""
