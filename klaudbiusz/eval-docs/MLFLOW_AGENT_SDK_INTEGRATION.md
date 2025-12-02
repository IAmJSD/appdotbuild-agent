# MLflow Agent SDK Integration

**Status:** Proposed
**Date:** 2025-01-18
**Context:** Tighter integration between Klaudbiusz evaluation framework and Databricks Agent SDK MLflow capabilities

## Overview

This document outlines how to integrate the 9-metric evaluation framework with Databricks Agent SDK's GenAI evaluation and monitoring features. The goal is to leverage native MLflow capabilities while maintaining the existing 90% app success rate.

**Key Resources:**
- [Databricks MLflow GenAI Eval & Monitor Docs](https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/)
- Current implementation: `cli/evaluate_all.py`, `cli/mlflow_tracker.py`

---

## Current State

### Existing Klaudbiusz Evaluation Framework

**9 Objective Metrics:**
1. Build Success (binary)
2. Runtime Success (binary)
3. Type Safety (binary)
4. Tests Pass (binary + coverage %)
5. Databricks Connectivity (binary)
6. Data Returned (binary)
7. UI Renders (binary, VLM check)
8. Local Runability (0-5 score)
9. Deployability (0-5 score)

**Composite Metrics:**
- `appeval_100`: Weighted score combining all metrics (0-100 scale)
- `eff_units`: Efficiency metric (tokens/1000 + turns + validation_runs)

**Current MLflow Integration:**
- Logs parameters, metrics, artifacts to Databricks MLflow
- Creates per-app table with all metrics (`app_metrics.json`)
- Logs trajectory files for each app
- Supports staging and production experiments

**Implementation:** `cli/evaluate_all.py`, `cli/eval_metrics.py`, `cli/mlflow_tracker.py`

---

## Integration Opportunities

### 1. Custom MLflow Judges for 9 Metrics

**Concept:** Convert objective metrics into reusable MLflow custom scorers that work across dev/test/prod.

**Implementation:**

```python
# cli/mlflow_judges.py (NEW FILE)
from mlflow.metrics import make_custom_scorer
from pathlib import Path
import subprocess

def type_safety_judge(trace_data):
    """Judge that checks TypeScript type safety."""
    app_dir = Path(trace_data.get("app_dir"))

    try:
        result = subprocess.run(
            ["npx", "tsc", "--noEmit"],
            cwd=app_dir,
            capture_output=True,
            timeout=60
        )
        passed = result.returncode == 0
        return {"score": 1.0 if passed else 0.0, "justification": result.stderr.decode()}
    except Exception as e:
        return {"score": 0.0, "justification": str(e)}

type_safety_metric = make_custom_scorer(
    name="type_safety",
    definition="Checks if TypeScript compilation passes with zero errors",
    judge_function=type_safety_judge,
    greater_is_better=True
)

def build_success_judge(trace_data):
    """Judge that checks Docker build success."""
    # Similar implementation
    pass

def databricks_connectivity_judge(trace_data):
    """Judge that checks Databricks connection."""
    # Similar implementation
    pass

# Export all judges
ALL_JUDGES = [
    type_safety_metric,
    build_success_metric,
    runtime_success_metric,
    tests_pass_metric,
    databricks_connectivity_metric,
    # ... remaining metrics
]
```

**Benefits:**
- DRY: Same judges in development, testing, production
- Consistent evaluation across lifecycle
- Native MLflow integration with tracing
- Easier to maintain and version

**Migration Path:**
1. Create `cli/mlflow_judges.py`
2. Implement judges for binary metrics first (Metrics 1-7)
3. Add composite judges for DevX scores (Metrics 8-9)
4. Update `evaluate_all.py` to optionally use judges via feature flag

---

### 2. Dataset-Driven Evaluation from Trajectories

**Concept:** Transform trajectory.jsonl files into MLflow evaluation datasets for systematic comparison.

**Current State:**
- Already logging trajectory files (evaluate_all.py:991-1008)
- Stored per-app at `app_dir/trajectory.jsonl`
- Not currently used for evaluation

**Implementation:**

```python
# cli/create_eval_dataset.py (NEW FILE)
import mlflow
import pandas as pd
from pathlib import Path

def create_dataset_from_trajectories(app_dirs: list[Path], prompts: dict) -> pd.DataFrame:
    """Create evaluation dataset from existing trajectories."""
    records = []

    for app_dir in app_dirs:
        trajectory_file = app_dir / "trajectory.jsonl"
        if not trajectory_file.exists():
            continue

        # Parse trajectory
        trajectory = parse_trajectory(trajectory_file)

        records.append({
            "app_name": app_dir.name,
            "prompt": prompts.get(app_dir.name),
            "trajectory": trajectory,
            "app_dir": str(app_dir),
            # Add ground truth if available
            "expected_appeval_100": None,  # Can be populated from previous runs
        })

    return pd.DataFrame(records)

def evaluate_with_dataset(dataset_df: pd.DataFrame):
    """Run MLflow evaluation with dataset."""
    # Create MLflow dataset
    dataset = mlflow.data.from_pandas(
        dataset_df,
        source="klaudbiusz-trajectories",
        name="klaudbiusz-eval-dataset",
        targets="expected_appeval_100"
    )

    # Evaluate using custom judges
    from mlflow_judges import ALL_JUDGES

    results = mlflow.evaluate(
        data=dataset,
        model_type="databricks-agent",
        evaluators=ALL_JUDGES,
        evaluator_config={
            "timeout": 600,  # 10 min per app
        }
    )

    return results
```

**Benefits:**
- Build evaluation dataset incrementally
- Track dataset versions alongside model/MCP versions
- Compare runs systematically (A/B testing)
- Reproducible evaluations

**Usage:**

```python
# In evaluate_all.py
dataset_df = create_dataset_from_trajectories(app_dirs, prompts)
results = evaluate_with_dataset(dataset_df)
```

---

### 3. Production Monitoring for Deployed Apps

**Concept:** Enable automated quality monitoring for deployed Databricks apps using scheduled scorers.

**Implementation:**

Add tracing to generated app templates:

```typescript
// edda_templates/trpc_react/server/index.ts (MODIFY)
import { trace } from '@databricks/sdk-agent-tracing';

app.use('/trpc', trpcExpress.createExpressMiddleware({
  router: appRouter,
  createContext: async ({ req }) => {
    const traceId = req.headers['x-request-id'] || generateId();

    // Start MLflow trace
    const span = trace.startSpan({
      name: req.path,
      attributes: {
        app_name: process.env.DATABRICKS_APP_NAME,
        request_id: traceId,
      }
    });

    return { traceId, span };
  },
}));
```

Schedule monitoring in evaluate_all.py:

```python
# cli/evaluate_all.py (ADD)
def setup_production_monitoring(tracker: EvaluationTracker):
    """Setup continuous monitoring for deployed apps."""
    if not tracker.enabled:
        return

    from mlflow_judges import ALL_JUDGES

    for judge in ALL_JUDGES:
        # Schedule judge to run on 10% of production traffic
        mlflow.schedule_scorer(
            scorer=judge,
            sample_rate=0.1,
            schedule="hourly",
            tags={"environment": "production"}
        )

    print("✓ Production monitoring scheduled")
```

**Monitored Metrics:**
- **Metric 2 (runtime_success):** Track as uptime/availability
- **Metric 5 (databricks_connectivity):** Monitor DB connection health
- **appeval_100:** Overall quality drift detection

**Benefits:**
- Catch regressions in production
- Early warning for quality degradation
- Automated alerts via Databricks workflows

---

### 4. Enhanced MLflow Table with Trace Linkage

**Concept:** Link each app evaluation to its generation trace for drill-down analysis.

**Current Implementation:**
```python
# cli/mlflow_tracker.py:227
mlflow.log_table(df, "app_metrics.json")
```

**Enhanced Implementation:**

```python
# cli/mlflow_tracker.py (MODIFY log_evaluation_metrics)
def log_evaluation_metrics(self, evaluation_report: Dict[str, Any]):
    """Log evaluation metrics with trace linkage."""
    # ... existing code ...

    app_records = []
    trace_ids = []

    for app in apps:
        # Extract trace ID from trajectory file
        trajectory_file = Path(app.get('app_dir', '')) / 'trajectory.jsonl'
        trace_id = extract_trace_id(trajectory_file) if trajectory_file.exists() else None

        record = {
            'app_name': app['app_name'],
            'trace_id': trace_id,  # NEW: Enable drill-down
            'request_id': app.get('generation_request_id'),  # NEW
            'generation_timestamp': app.get('timestamp'),
            # ... existing metrics ...
        }
        app_records.append(record)
        if trace_id:
            trace_ids.append(trace_id)

    # Log table with trace metadata
    if app_records:
        df = pd.DataFrame(app_records)
        mlflow.log_table(
            df,
            "app_metrics.json",
            metadata={"trace_ids": trace_ids}  # Enable trace queries
        )

def extract_trace_id(trajectory_file: Path) -> str | None:
    """Extract trace ID from trajectory JSONL."""
    try:
        with open(trajectory_file) as f:
            first_line = f.readline()
            data = json.loads(first_line)
            return data.get('trace_id')
    except:
        return None
```

**Benefits:**
- Click from metric → full generation trace in Databricks UI
- Root cause analysis for failures
- Link evaluation results to generation context
- Query: "Show me all apps where appeval_100 < 70 and their traces"

---

### 5. Human Feedback Integration

**Concept:** Add human review workflow for failed apps using Databricks Review App.

**Implementation:**

```python
# cli/request_human_review.py (NEW FILE)
from mlflow.tracking import MlflowClient

def request_reviews_for_failing_apps(evaluation_report: dict, run_id: str):
    """Create review requests for apps below quality threshold."""
    client = MlflowClient()

    apps = evaluation_report.get('apps', [])
    review_threshold = 70  # appeval_100 < 70

    for app in apps:
        appeval_score = app['metrics'].get('appeval_100', 0)

        if appeval_score < review_threshold:
            # Tag for review
            client.set_tag(
                run_id=run_id,
                key=f"review_requested.{app['app_name']}",
                value="true"
            )

            # Log review context
            review_context = {
                "app_name": app['app_name'],
                "appeval_100": appeval_score,
                "issues": app.get('issues', []),
                "prompt": app.get('prompt', ''),
                "trace_id": app.get('trace_id'),
            }

            client.log_dict(
                run_id=run_id,
                dictionary=review_context,
                artifact_file=f"reviews/{app['app_name']}_context.json"
            )

    print(f"✓ Review requests created for {len([a for a in apps if a['metrics'].get('appeval_100', 0) < review_threshold])} apps")
```

**Review Questions:**
- Are the 9 metrics correctly measuring quality?
- What's the root cause of this failure?
- Should this be flagged as a template issue or generation issue?
- Quality label: excellent/good/fair/poor

**Integration with evaluate_all.py:**

```python
# After MLflow logging
if tracker.enabled:
    request_reviews_for_failing_apps(full_report, run_id)
```

---

### 6. Concrete Implementation Plan

**Phase 1: Convert Metrics to Judges** (1-2 days)
- [ ] Create `cli/mlflow_judges.py`
- [ ] Implement judges for Metrics 1-7 (binary metrics)
- [ ] Add composite judges for Metrics 8-9 (DevX scores)
- [ ] Add appeval_100 composite judge
- [ ] Test with single app: `pytest cli/test_mlflow_judges.py`
- [ ] Add feature flag to `evaluate_all.py`: `--use-mlflow-judges`

**Phase 2: Dataset Integration** (1 day)
- [ ] Create `cli/create_eval_dataset.py`
- [ ] Implement `create_dataset_from_trajectories()`
- [ ] Implement `evaluate_with_dataset()`
- [ ] Test dataset creation from existing trajectories
- [ ] Add feature flag: `--use-dataset-evaluation`
- [ ] Compare results with current framework (should match)

**Phase 3: Trace Linkage** (1 day)
- [ ] Add `extract_trace_id()` to `mlflow_tracker.py`
- [ ] Modify `log_evaluation_metrics()` to include trace_id
- [ ] Test trace linkage in Databricks UI
- [ ] Verify drill-down from metrics to traces works

**Phase 4: Production Monitoring** (2-3 days)
- [ ] Add tracing to `edda_templates/trpc_react/server/index.ts`
- [ ] Install `@databricks/sdk-agent-tracing` in template
- [ ] Create `setup_production_monitoring()` function
- [ ] Deploy test app and verify traces appear
- [ ] Configure scheduled scorers (Beta feature)
- [ ] Set up Databricks alerts for appeval_100 drops

**Phase 5: Human Feedback Loop** (1 day)
- [ ] Create `cli/request_human_review.py`
- [ ] Integrate with `evaluate_all.py`
- [ ] Test review request creation
- [ ] Document review workflow for team

**Phase 6: Documentation & Cleanup** (1 day)
- [ ] Update `eval-docs/evals.md` with new features
- [ ] Add migration guide for existing evaluations
- [ ] Update `README.md` with MLflow SDK features
- [ ] Add examples to docs

---

## Migration Strategy

**Hybrid Approach:** Keep existing framework working while gradually adding SDK features.

```python
# cli/evaluate_all.py
def parse_args():
    parser.add_argument(
        '--use-mlflow-judges',
        action='store_true',
        help='Use MLflow custom judges instead of direct evaluation'
    )
    parser.add_argument(
        '--use-dataset-evaluation',
        action='store_true',
        help='Use MLflow dataset-driven evaluation'
    )
    return parser.parse_args()

async def main_async():
    args = parse_args()

    if args.use_mlflow_judges and args.use_dataset_evaluation:
        # New: MLflow SDK evaluation
        dataset_df = create_dataset_from_trajectories(app_dirs, prompts)
        results = evaluate_with_dataset(dataset_df)
    else:
        # Current: Direct Dagger evaluation
        results = await evaluate_app_async(...)
```

**Validation:**
1. Run both old and new evaluation on same dataset
2. Compare results (should be identical for binary metrics)
3. If discrepancies found, investigate before full migration
4. Gradual rollout: 10% → 50% → 100%

---

## Expected Benefits

### Efficiency Gains

**Before Integration:**
- Custom evaluation framework ✓
- Manual MLflow logging ✓
- Separate production monitoring ✗
- Manual trace analysis ✗

**After Integration:**
- Reuse same judges everywhere (DRY principle)
- Automatic trace-to-evaluation linkage
- Production quality monitoring out-of-the-box
- Dataset versioning for reproducibility
- Human feedback loop integrated

### Cost Impact

**Additional Costs:**
- MLflow storage for datasets: ~$0.01/GB/month (minimal)
- Production monitoring: ~$0.001/request * sample_rate (configurable)

**Cost Savings:**
- Reduced manual investigation time (trace linkage)
- Earlier detection of quality issues (production monitoring)
- Reduced rework from human feedback loop

### Quality Impact

**Expected Improvements:**
- Faster root cause analysis (trace linkage)
- Proactive issue detection (production monitoring)
- Better ground truth labels (human feedback)
- More systematic A/B testing (dataset versioning)

**Risks:**
- Integration complexity (mitigated by phased approach)
- Beta feature stability (production monitoring)
- Team learning curve (new SDK concepts)

---

## Success Metrics

Track these metrics to validate integration success:

1. **Time to Root Cause:** Measure time from issue detection to root cause identification
   - Target: 50% reduction with trace linkage

2. **Production Issue Detection Rate:** % of issues caught before user reports
   - Target: 80% with scheduled monitoring

3. **Evaluation Consistency:** Variance between dev and prod evaluations
   - Target: <5% difference with shared judges

4. **Dataset Growth:** Number of evaluation examples collected
   - Target: 100+ examples within 3 months

5. **Human Review Efficiency:** Time spent on low-quality app reviews
   - Target: Focus 80% of review time on apps with appeval_100 < 70

---

## Next Steps

**Immediate (This Week):**
1. Review this document with team
2. Decide on Phase 1 priority (Custom Judges)
3. Set up feature branch: `git checkout -b feat/mlflow-agent-sdk-integration`

**Short Term (Next Sprint):**
1. Implement Phase 1: Custom MLflow Judges
2. Implement Phase 3: Trace Linkage
3. Test with 5-10 apps, compare with existing framework

**Medium Term (Next Month):**
1. Complete Phase 2: Dataset Integration
2. Complete Phase 5: Human Feedback Loop
3. Run parallel evaluations (old + new) for validation

**Long Term (Next Quarter):**
1. Complete Phase 4: Production Monitoring
2. Migrate all evaluations to new framework
3. Deprecate custom evaluation code (if appropriate)

---

## Meta-Agentic Feedback Loop: Self-Improving Code Generation

**Key Insight:** The generated Databricks apps are themselves agentic (they query data, make decisions). By instrumenting them with MLflow Tracing, we can create a feedback loop where production app behavior improves code generation.

### Architecture: Three-Level Tracing

```
Level 1: Code Generation Tracing
  └─ klaudbiusz agent generates app code
  └─ Logs: prompts, tool calls, validation results
  └─ Trace ID: gen_trace_12345

Level 2: Generated App Runtime Tracing
  └─ Customer-churn-analysis app runs in production
  └─ Logs: tRPC calls, Databricks queries, user interactions
  └─ Trace ID: app_trace_67890
  └─ Links to: gen_trace_12345 (which generation created this app)

Level 3: Evaluation & Improvement Tracing
  └─ Scorers analyze both gen and app traces
  └─ Feed results back to klaudbiusz for next generation
  └─ Trace ID: eval_trace_11111
  └─ Links to: gen_trace_12345 + app_trace_67890
```

### Implementation: Instrumenting Generated Apps

**Step 1: Add MLflow Tracing to Template**

Modify `edda_templates/template_trpc/server/src/index.ts`:

```typescript
import { initTRPC } from "@trpc/server";
import { createExpressMiddleware } from "@trpc/server/adapters/express";
import express from "express";
import "dotenv/config";
import superjson from "superjson";
import path from "node:path";
import * as mlflow from 'mlflow-tracing';

// Initialize MLflow Tracing (only if credentials are set)
if (process.env.DATABRICKS_TOKEN && process.env.DATABRICKS_HOST) {
  mlflow.init({
    trackingUri: 'databricks',
    experimentId: process.env.MLFLOW_EXPERIMENT_ID || undefined,
  });
  console.log('✓ MLflow Tracing enabled');
}

const STATIC_DIR = path.join(__dirname, "..", "public");

const t = initTRPC.create({
  transformer: superjson,
});

const publicProcedure = t.procedure;
const router = t.router;

// Middleware to add tracing to tRPC procedures
const tracedProcedure = publicProcedure.use(async (opts) => {
  // Extract generation trace ID from environment
  const generationTraceId = process.env.GENERATION_TRACE_ID;

  return await mlflow.withSpan(
    async (span) => {
      // Link to generation trace
      if (generationTraceId) {
        span.setTag('generation_trace_id', generationTraceId);
      }

      // Add app metadata
      span.setTag('app_name', process.env.DATABRICKS_APP_NAME || 'unknown');
      span.setTag('procedure', opts.path);

      try {
        const result = await opts.next();
        span.end({ outputs: { success: true }, status: 'OK' });
        return result;
      } catch (error) {
        span.end({
          outputs: { success: false, error: String(error) },
          status: 'ERROR'
        });
        throw error;
      }
    },
    {
      name: `trpc.${opts.path}`,
      spanType: mlflow.SpanType.TOOL
    }
  );
});

export const appRouter = router({
  healthcheck: tracedProcedure.query(() => {
    return { status: "ok", timestamp: new Date().toISOString() };
  }),

  // Example: Traced Databricks query
  getChurnAnalysis: tracedProcedure.query(async () => {
    return await mlflow.withSpan(
      async (span) => {
        const client = new DatabricksClient();
        const query = `SELECT * FROM customer_churn LIMIT 100`;

        span.setInput('query', query);

        const { rows } = await client.executeQuery(query);

        span.setOutput('row_count', rows.length);
        span.setTag('table', 'customer_churn');

        return rows;
      },
      {
        name: 'databricks.query',
        spanType: mlflow.SpanType.TOOL
      }
    );
  }),
});

// ... rest of server code
```

**Step 2: Update package.json Dependencies**

```json
{
  "dependencies": {
    "mlflow-tracing": "^3.6.0"
  }
}
```

**Step 3: Pass Generation Trace ID to Generated App**

Modify `cli/codegen.py` to inject trace ID:

```python
# cli/codegen.py (NEW SECTION)
def inject_generation_metadata(app_dir: Path, trace_id: str):
    """Inject generation trace ID into app environment."""
    env_file = app_dir / ".env"

    with open(env_file, 'a') as f:
        f.write(f"\n# Generation metadata\n")
        f.write(f"GENERATION_TRACE_ID={trace_id}\n")
        f.write(f"MLFLOW_EXPERIMENT_ID=/Shared/klaudbiusz-app-monitoring\n")
```

### Runtime Scorers: Evaluating Production App Behavior

Create scorers that work with **live trace data** instead of static code:

```python
# cli/mlflow_runtime_scorers.py (NEW FILE)
from mlflow.metrics import make_custom_scorer
from mlflow.tracking import MlflowClient

def query_efficiency_scorer(trace_data):
    """
    Scorer that evaluates Databricks query efficiency from production traces.

    Checks:
    - Query execution time
    - Result set size
    - Use of warehouse caching
    """
    client = MlflowClient()

    # Extract app trace ID from trace_data
    app_trace_id = trace_data.get('trace_id')

    # Get all databricks.query spans from this trace
    trace = client.get_trace(app_trace_id)
    query_spans = [s for s in trace.spans if s.name == 'databricks.query']

    scores = []
    justifications = []

    for span in query_spans:
        duration_ms = span.duration_ms
        row_count = span.outputs.get('row_count', 0)

        # Score based on efficiency
        if duration_ms < 1000 and row_count < 10000:
            scores.append(1.0)
            justifications.append(f"✓ Efficient query: {duration_ms}ms, {row_count} rows")
        elif duration_ms < 5000:
            scores.append(0.7)
            justifications.append(f"⚠ Moderate query: {duration_ms}ms, {row_count} rows")
        else:
            scores.append(0.3)
            justifications.append(f"✗ Slow query: {duration_ms}ms, {row_count} rows")

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "score": avg_score,
        "justification": "\n".join(justifications),
        "metadata": {
            "query_count": len(query_spans),
            "avg_duration_ms": sum(s.duration_ms for s in query_spans) / len(query_spans) if query_spans else 0
        }
    }

def error_rate_scorer(trace_data):
    """Scorer that checks error rate in production."""
    client = MlflowClient()
    trace = client.get_trace(trace_data.get('trace_id'))

    total_spans = len(trace.spans)
    error_spans = [s for s in trace.spans if s.status == 'ERROR']
    error_rate = len(error_spans) / total_spans if total_spans > 0 else 0

    score = 1.0 - error_rate

    return {
        "score": score,
        "justification": f"Error rate: {error_rate*100:.1f}% ({len(error_spans)}/{total_spans} spans)",
        "metadata": {
            "error_spans": [s.name for s in error_spans]
        }
    }

def user_experience_scorer(trace_data):
    """Scorer that evaluates end-to-end user experience."""
    client = MlflowClient()
    trace = client.get_trace(trace_data.get('trace_id'))

    # Root span is the user request
    root_span = [s for s in trace.spans if s.parent_id is None][0]
    total_duration_ms = root_span.duration_ms

    # Score based on response time
    if total_duration_ms < 500:
        score = 1.0
        justification = f"✓ Excellent UX: {total_duration_ms}ms"
    elif total_duration_ms < 2000:
        score = 0.8
        justification = f"Good UX: {total_duration_ms}ms"
    elif total_duration_ms < 5000:
        score = 0.5
        justification = f"⚠ Moderate UX: {total_duration_ms}ms"
    else:
        score = 0.2
        justification = f"✗ Poor UX: {total_duration_ms}ms"

    return {
        "score": score,
        "justification": justification,
        "metadata": {
            "duration_ms": total_duration_ms
        }
    }

# Register runtime scorers
RUNTIME_SCORERS = [
    make_custom_scorer(
        name="query_efficiency",
        definition="Evaluates Databricks query efficiency from production traces",
        judge_function=query_efficiency_scorer,
        greater_is_better=True
    ),
    make_custom_scorer(
        name="error_rate",
        definition="Checks production error rate",
        judge_function=error_rate_scorer,
        greater_is_better=True
    ),
    make_custom_scorer(
        name="user_experience",
        definition="Evaluates end-to-end user experience timing",
        judge_function=user_experience_scorer,
        greater_is_better=True
    ),
]
```

### Production Monitoring Setup

Schedule runtime scorers to continuously evaluate production apps:

```python
# cli/setup_production_monitoring.py (NEW FILE)
import mlflow
from mlflow_runtime_scorers import RUNTIME_SCORERS

def setup_monitoring_for_app(app_name: str, sample_rate: float = 0.1):
    """
    Setup production monitoring for a deployed app.

    Args:
        app_name: Name of the deployed app
        sample_rate: % of requests to evaluate (0.0-1.0)
    """
    experiment_name = f"/Shared/klaudbiusz-app-monitoring/{app_name}"
    mlflow.set_experiment(experiment_name)

    for scorer in RUNTIME_SCORERS:
        mlflow.schedule_scorer(
            scorer=scorer,
            filter={"app_name": app_name},
            sample_rate=sample_rate,
            schedule="hourly",
            tags={
                "environment": "production",
                "app_name": app_name
            }
        )

    print(f"✓ Production monitoring enabled for {app_name}")
    print(f"  Sample rate: {sample_rate*100}%")
    print(f"  Experiment: {experiment_name}")

# Usage
if __name__ == "__main__":
    # Setup monitoring for all deployed apps
    deployed_apps = [
        "customer-churn-analysis",
        "product-profitability",
        "revenue-forecast-quarterly"
    ]

    for app in deployed_apps:
        setup_monitoring_for_app(app, sample_rate=0.1)
```

### Meta-Agentic Feedback Loop

Create a feedback system that uses evaluation results to improve future code generation:

```python
# cli/improvement_loop.py (NEW FILE)
"""
Meta-agentic improvement loop that analyzes evaluation results
and generates improvement suggestions for the code generator.
"""

from pathlib import Path
import json
from mlflow.tracking import MlflowClient
from anthropic import Anthropic

def analyze_generation_patterns(evaluation_results: dict) -> dict:
    """
    Analyze patterns across evaluations to identify systemic issues.

    Returns:
        {
            "common_failures": [...],
            "anti_patterns": [...],
            "improvement_suggestions": [...]
        }
    """
    apps = evaluation_results.get('apps', [])

    # Group by failure modes
    failure_modes = defaultdict(list)
    for app in apps:
        if app['metrics']['appeval_100'] < 70:
            for issue in app.get('issues', []):
                failure_modes[issue].append(app['app_name'])

    # Identify patterns
    common_failures = [
        {"issue": issue, "count": len(apps), "examples": apps[:3]}
        for issue, apps in failure_modes.items()
        if len(apps) >= 3  # At least 3 occurrences
    ]

    return {
        "common_failures": common_failures,
        "total_apps": len(apps),
        "avg_quality": sum(a['metrics']['appeval_100'] for a in apps) / len(apps)
    }

def generate_improvement_prompt(analysis: dict, generation_trace_id: str) -> str:
    """
    Generate a prompt for the klaudbiusz agent to improve its code generation.

    Uses evaluation insights to create specific improvement instructions.
    """
    prompt = f"""# Code Generation Improvement Analysis

Based on evaluation of {analysis['total_apps']} generated apps:

**Current Quality Score:** {analysis['avg_quality']:.1f}/100

## Common Failure Patterns

"""

    for failure in analysis['common_failures']:
        prompt += f"\n### {failure['issue']}\n"
        prompt += f"- Frequency: {failure['count']}/{analysis['total_apps']} apps\n"
        prompt += f"- Examples: {', '.join(failure['examples'])}\n"

    prompt += """

## Improvement Task

Review the generation trace (ID: {generation_trace_id}) and the common failures above.

Generate specific improvements to the code generation templates and prompts to address these issues:

1. Identify root cause in generation logic
2. Propose concrete template/prompt changes
3. Create validation checks to prevent recurrence

Output format: JSON with fields:
- root_cause: string
- proposed_changes: array of {file: string, change: string}
- validation_rules: array of {rule: string, check: string}
"""

    return prompt.format(generation_trace_id=generation_trace_id)

def apply_improvements(improvement_suggestions: dict):
    """
    Apply improvements to code generation templates and agent prompts.

    This could be:
    1. Automated (direct file edits)
    2. Semi-automated (PR creation)
    3. Manual (GitHub issue creation)
    """
    # For safety, start with manual approach (create GitHub issues)
    for suggestion in improvement_suggestions['proposed_changes']:
        create_github_issue(
            title=f"Improve: {suggestion['file']}",
            body=f"""
## Root Cause
{improvement_suggestions['root_cause']}

## Proposed Change
{suggestion['change']}

## Validation
- [ ] {suggestion.get('validation_rule', 'Manual testing')}
            """,
            labels=["codegen-improvement", "auto-generated"]
        )

def run_improvement_loop():
    """
    Main improvement loop:
    1. Load latest evaluation results
    2. Analyze patterns
    3. Generate improvement prompt
    4. Use Claude to generate improvements
    5. Create issues/PRs for implementation
    """
    # Load evaluation results
    eval_report_path = Path(__file__).parent.parent / "app-eval" / "evaluation_report.json"
    with open(eval_report_path) as f:
        evaluation_results = json.load(f)

    # Analyze patterns
    analysis = analyze_generation_patterns(evaluation_results)

    if not analysis['common_failures']:
        print("✓ No systemic issues found! Quality is high.")
        return

    print(f"Found {len(analysis['common_failures'])} systemic issues")

    # Get generation trace ID (from most recent run)
    # This would come from bulk_run.py logging
    generation_trace_id = "get_from_mlflow"

    # Generate improvement prompt
    prompt = generate_improvement_prompt(analysis, generation_trace_id)

    # Use Claude to analyze and suggest improvements
    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    suggestions = json.loads(response.content[0].text)

    # Apply improvements (create GitHub issues)
    apply_improvements(suggestions)

    print(f"✓ Created {len(suggestions['proposed_changes'])} improvement issues")

if __name__ == "__main__":
    run_improvement_loop()
```

### Closed-Loop System: From Generation to Production to Improvement

**Complete Flow:**

```
1. CODE GENERATION (Level 1 Tracing)
   ├─ User: "Create customer churn dashboard"
   ├─ klaudbiusz agent generates code
   ├─ MLflow logs generation trace with:
   │  ├─ Prompt
   │  ├─ Tool calls (scaffold, databricks queries)
   │  ├─ Validation results
   │  └─ Generation trace ID: gen_abc123
   └─ Output: customer-churn-analysis app

2. STATIC EVALUATION (Current System)
   ├─ evaluate_all.py runs 9 metrics
   ├─ Links results to gen_abc123
   ├─ appeval_100 score: 85/100
   └─ Identifies: "TypeScript errors in 3 files"

3. PRODUCTION DEPLOYMENT
   ├─ App deployed with GENERATION_TRACE_ID=gen_abc123
   ├─ App instruments itself with MLflow Tracing
   └─ Every tRPC call creates spans linked to gen_abc123

4. RUNTIME EVALUATION (Level 2 Tracing)
   ├─ Scheduled scorers run on production traces
   ├─ Query efficiency scorer: 0.95/1.0 ✓
   ├─ Error rate scorer: 0.88/1.0 (12% errors)
   ├─ User experience scorer: 0.92/1.0 ✓
   └─ Issue found: "Slow query on customer_orders table"

5. META-AGENTIC IMPROVEMENT (Level 3 Tracing)
   ├─ improvement_loop.py analyzes patterns across all apps
   ├─ Finds: 8/20 apps have slow customer queries
   ├─ Traces back to generation template
   ├─ Claude suggests: "Add LIMIT clause to all customer queries"
   ├─ Creates GitHub issue with proposed fix
   └─ Human reviews and merges improvement

6. NEXT GENERATION (Improved)
   ├─ User: "Create customer retention dashboard"
   ├─ klaudbiusz uses improved template
   ├─ Automatically adds LIMIT clauses
   ├─ appeval_100 score: 92/100 ✓
   └─ Runtime scorers: 0.98/1.0 ✓
```

### Benefits of Three-Level Tracing

**Development Phase:**
- Understand what generation patterns lead to high-quality apps
- Debug why certain apps fail validation
- Compare different agent models/prompts

**Production Phase:**
- Monitor real user behavior and performance
- Catch issues that static evaluation missed
- Validate that apps work correctly with real data

**Improvement Phase:**
- Identify systemic issues across all apps
- Data-driven template improvements
- Continuous quality improvement loop

### Implementation Timeline

**Phase 1: Instrument Generated Apps** (2-3 days)
- Add mlflow-tracing to template dependencies
- Implement tracedProcedure middleware
- Test with one deployed app
- Verify traces appear in Databricks

**Phase 2: Runtime Scorers** (2 days)
- Implement query_efficiency_scorer
- Implement error_rate_scorer
- Implement user_experience_scorer
- Schedule on production apps (10% sample)

**Phase 3: Link Generation to Runtime** (1 day)
- Inject GENERATION_TRACE_ID in codegen
- Update evaluate_all.py to include runtime metrics
- Create unified dashboard showing both static + runtime scores

**Phase 4: Meta-Agentic Loop** (3-4 days)
- Implement pattern analysis (improvement_loop.py)
- Create GitHub issue automation
- Test improvement cycle end-to-end
- Document feedback loop process

**Phase 5: Continuous Improvement** (Ongoing)
- Monitor improvement suggestions weekly
- Review and apply high-value improvements
- Track quality metrics over time
- Iterate on scorer definitions

### Success Metrics for Meta-Agentic Loop

Track these to validate the improvement loop works:

1. **Quality Improvement Rate**
   - Metric: Δ appeval_100 per generation cohort
   - Target: +5 points per month

2. **Issue Reduction Rate**
   - Metric: % decrease in common_failures count
   - Target: -20% per quarter

3. **Runtime Performance**
   - Metric: Average runtime scorer scores
   - Target: >0.90 across all scorers

4. **Suggestion Quality**
   - Metric: % of improvement suggestions accepted
   - Target: >60% accepted and merged

5. **Time to Fix**
   - Metric: Days from issue detection to fix deployment
   - Target: <7 days for systemic issues

---

## Publishing Generated Apps and Daily Evaluation Pipeline

**Status:** Proposed
**Priority:** High
**Context:** Enable continuous quality monitoring and public transparency by publishing generated app sources and running automated daily evaluations.

### Goals

1. **Transparency:** Make generated apps publicly accessible for inspection and reproduction
2. **Continuous Quality Assurance:** Catch regressions early via automated daily evaluations
3. **Benchmark Dataset:** Build a growing dataset of evaluated apps for research and improvement
4. **Community Feedback:** Enable external developers to report issues and suggest improvements

---

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Daily Evaluation Pipeline                 │
└─────────────────────────────────────────────────────────────┘

1. SCHEDULED TRIGGER (Daily @ 2am UTC)
   └─ GitHub Actions / Databricks Workflow
   └─ Triggers: bulk_run.py with daily prompts

2. BATCH GENERATION
   ├─ Reads prompts from: cli/prompts/daily_prompts.json
   ├─ Generates 5-10 apps per day
   ├─ Logs to MLflow: gen_trace_[date]
   └─ Outputs: app/[app-name]-[date]/

3. EVALUATION
   ├─ Runs: evaluate_all.py --mlflow-enabled
   ├─ Executes all 9 metrics
   ├─ Logs results to MLflow experiment: /Shared/klaudbiusz-daily-evals
   └─ Generates: evaluation_report_[date].json

4. PUBLISHING
   ├─ Archives app sources → GitHub release
   ├─ Uploads artifacts → MLflow
   ├─ Updates public dashboard
   └─ Sends alerts if quality drops

5. CLEANUP
   ├─ Stops running containers
   ├─ Archives logs
   └─ Prunes old Docker images
```

---

### 1. Publishing Generated App Sources

#### 1.1 GitHub Repository Structure

Create a dedicated public repository for generated apps:

**Repository:** `anthropic/klaudbiusz-generated-apps`

```
klaudbiusz-generated-apps/
├── README.md                          # Overview, links to dashboard
├── apps/
│   ├── 2025-01-20/                   # Daily batch
│   │   ├── customer-churn-analysis/
│   │   │   ├── src/
│   │   │   ├── package.json
│   │   │   ├── Dockerfile
│   │   │   ├── .env.example
│   │   │   └── GENERATION.md         # Prompt, trace ID, metrics
│   │   ├── product-profitability/
│   │   └── revenue-forecast/
│   └── 2025-01-21/
├── evaluations/
│   ├── 2025-01-20_report.json        # Full evaluation report
│   ├── 2025-01-20_summary.md         # Human-readable summary
│   └── metrics_timeseries.csv        # Historical metrics
├── .github/
│   └── workflows/
│       └── daily-eval.yml            # CI workflow
└── docs/
    ├── METRICS.md                     # Metrics documentation
    └── REPRODUCTION.md                # How to run apps locally
```

#### 1.2 Automated Publishing Script

```python
# cli/publish_apps.py (NEW FILE)
"""
Publish generated apps to public GitHub repository.
"""
import shutil
from pathlib import Path
from datetime import date
import subprocess
import json

PUBLISH_REPO = Path("~/klaudbiusz-generated-apps").expanduser()

def prepare_app_for_publishing(app_dir: Path) -> dict:
    """
    Prepare app for public release.

    - Removes sensitive data (.env, secrets)
    - Creates .env.example
    - Adds GENERATION.md with metadata
    """
    # Copy app to temp location
    app_name = app_dir.name

    # Remove sensitive files
    sensitive_files = [".env", ".edda_state", "node_modules", ".next"]
    for filename in sensitive_files:
        sensitive_path = app_dir / filename
        if sensitive_path.exists():
            if sensitive_path.is_dir():
                shutil.rmtree(sensitive_path)
            else:
                sensitive_path.unlink()

    # Create .env.example from .env.template
    env_example = app_dir / ".env.example"
    if not env_example.exists():
        with open(env_example, 'w') as f:
            f.write("""# Databricks Configuration
DATABRICKS_HOST=https://your-workspace.databricks.com
DATABRICKS_TOKEN=dapi...
DATABRICKS_WAREHOUSE_ID=your-warehouse-id

# App Configuration
DATABRICKS_APP_NAME={app_name}
DATABRICKS_APP_PORT=3000
""".format(app_name=app_name))

    # Create GENERATION.md with metadata
    generation_md = app_dir / "GENERATION.md"
    metadata = load_app_metadata(app_dir)

    with open(generation_md, 'w') as f:
        f.write(f"""# Generation Metadata

**App Name:** {app_name}
**Generated:** {metadata.get('timestamp', 'N/A')}
**Prompt:** {metadata.get('prompt', 'N/A')}

## Quality Metrics

- **appeval_100:** {metadata.get('appeval_100', 'N/A')}/100
- **Build Success:** {'✓' if metadata.get('build_success') else '✗'}
- **Type Safety:** {'✓' if metadata.get('type_safety') else '✗'}
- **Tests Pass:** {'✓' if metadata.get('tests_pass') else '✗'}
- **Databricks Connected:** {'✓' if metadata.get('databricks_connectivity') else '✗'}

## MLflow Links

- **Generation Trace:** [{metadata.get('trace_id', 'N/A')}](https://your-workspace.databricks.com/ml/experiments/.../runs/{metadata.get('run_id', '')})
- **Evaluation Report:** [View Report](../../evaluations/{date.today()}_report.json)

## Reproduction

```bash
# Clone and run locally
cd {app_name}
cp .env.example .env
# Edit .env with your Databricks credentials

# Install and run
npm install
npm run dev
```

Visit http://localhost:3000
""")

    return metadata

def publish_daily_batch(apps: list[Path], evaluation_report: dict):
    """
    Publish a daily batch of apps to GitHub.
    """
    today = date.today().isoformat()

    # Create daily directory in publish repo
    daily_dir = PUBLISH_REPO / "apps" / today
    daily_dir.mkdir(parents=True, exist_ok=True)

    # Copy each app
    for app_dir in apps:
        if not app_dir.exists():
            continue

        print(f"Publishing {app_dir.name}...")

        # Prepare app
        metadata = prepare_app_for_publishing(app_dir)

        # Copy to publish location
        dest_dir = daily_dir / app_dir.name
        shutil.copytree(app_dir, dest_dir, dirs_exist_ok=True)

    # Copy evaluation report
    eval_dir = PUBLISH_REPO / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)

    report_file = eval_dir / f"{today}_report.json"
    with open(report_file, 'w') as f:
        json.dump(evaluation_report, f, indent=2)

    # Generate summary markdown
    summary_file = eval_dir / f"{today}_summary.md"
    generate_summary_markdown(evaluation_report, summary_file)

    # Update metrics timeseries
    update_metrics_timeseries(evaluation_report)

    # Commit and push
    commit_and_push(today, len(apps))

    print(f"✓ Published {len(apps)} apps for {today}")

def generate_summary_markdown(report: dict, output_file: Path):
    """Generate human-readable evaluation summary."""
    apps = report.get('apps', [])
    summary = report.get('summary', {})

    with open(output_file, 'w') as f:
        f.write(f"""# Evaluation Summary - {date.today()}

## Overview

- **Total Apps:** {len(apps)}
- **Average appeval_100:** {summary.get('avg_appeval_100', 0):.1f}/100
- **Success Rate:** {summary.get('success_rate', 0)*100:.1f}%

## Metrics Breakdown

| Metric | Pass Rate |
|--------|-----------|
| Build Success | {summary.get('build_success_rate', 0)*100:.0f}% |
| Type Safety | {summary.get('type_safety_rate', 0)*100:.0f}% |
| Tests Pass | {summary.get('tests_pass_rate', 0)*100:.0f}% |
| Databricks Connected | {summary.get('databricks_connectivity_rate', 0)*100:.0f}% |
| Data Returned | {summary.get('data_returned_rate', 0)*100:.0f}% |
| UI Renders | {summary.get('ui_renders_rate', 0)*100:.0f}% |

## Individual Apps

""")

        for app in apps:
            score = app['metrics'].get('appeval_100', 0)
            status = "✓" if score >= 70 else "✗"
            f.write(f"- {status} **{app['app_name']}** - {score:.0f}/100\n")

def update_metrics_timeseries(report: dict):
    """Append daily metrics to timeseries CSV."""
    metrics_file = PUBLISH_REPO / "evaluations" / "metrics_timeseries.csv"

    summary = report.get('summary', {})
    today = date.today().isoformat()

    # Create or append to CSV
    if not metrics_file.exists():
        with open(metrics_file, 'w') as f:
            f.write("date,avg_appeval_100,success_rate,build_success_rate,type_safety_rate,tests_pass_rate\n")

    with open(metrics_file, 'a') as f:
        f.write(f"{today},{summary.get('avg_appeval_100', 0):.2f},{summary.get('success_rate', 0):.2f},{summary.get('build_success_rate', 0):.2f},{summary.get('type_safety_rate', 0):.2f},{summary.get('tests_pass_rate', 0):.2f}\n")

def commit_and_push(date_str: str, num_apps: int):
    """Commit and push to GitHub."""
    subprocess.run(
        ["git", "add", "."],
        cwd=PUBLISH_REPO,
        check=True
    )

    subprocess.run(
        ["git", "commit", "-m", f"Daily evaluation {date_str} ({num_apps} apps)"],
        cwd=PUBLISH_REPO,
        check=True
    )

    subprocess.run(
        ["git", "push", "origin", "main"],
        cwd=PUBLISH_REPO,
        check=True
    )

if __name__ == "__main__":
    # Usage: python cli/publish_apps.py
    from pathlib import Path
    import json

    # Get apps from today's evaluation
    app_dir = Path(__file__).parent.parent / "app"
    apps = sorted(app_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]

    # Load evaluation report
    report_path = Path(__file__).parent.parent / "app-eval" / "evaluation_report.json"
    with open(report_path) as f:
        report = json.load(f)

    publish_daily_batch(apps, report)
```

#### 1.3 Artifact Publishing to MLflow

In addition to GitHub, publish artifacts to MLflow for long-term storage:

```python
# cli/mlflow_tracker.py (ADD METHOD)
def publish_app_artifacts(self, app_dir: Path):
    """Upload full app source as MLflow artifact."""
    if not self.enabled:
        return

    # Create tarball of app
    import tarfile

    tarball_path = app_dir.parent / f"{app_dir.name}.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(app_dir, arcname=app_dir.name)

    # Upload to MLflow
    mlflow.log_artifact(tarball_path, artifact_path="app_sources")

    # Cleanup
    tarball_path.unlink()
```

---

### 2. Daily Evaluation Pipeline

#### 2.1 GitHub Actions Workflow

```yaml
# .github/workflows/daily-eval.yml (NEW FILE)
name: Daily App Evaluation

on:
  schedule:
    # Run daily at 2am UTC
    - cron: '0 2 * * *'
  workflow_dispatch:  # Allow manual trigger

jobs:
  evaluate:
    runs-on: ubuntu-latest
    timeout-minutes: 180  # 3 hours max

    steps:
      - name: Checkout klaudbiusz
        uses: actions/checkout@v4
        with:
          repository: anthropic/agent
          path: agent

      - name: Checkout publish repo
        uses: actions/checkout@v4
        with:
          repository: anthropic/klaudbiusz-generated-apps
          path: publish
          token: ${{ secrets.PUBLISH_REPO_TOKEN }}

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install uv
        run: pip install uv

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Setup Docker
        uses: docker/setup-buildx-action@v3

      - name: Install dependencies
        working-directory: agent/klaudbiusz
        run: uv sync

      - name: Run daily generation
        working-directory: agent/klaudbiusz
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
          DATABRICKS_WAREHOUSE_ID: ${{ secrets.DATABRICKS_WAREHOUSE_ID }}
        run: |
          uv run cli/bulk_run.py \
            --prompts cli/prompts/daily_prompts.json \
            --max-apps 10 \
            --mode daily

      - name: Run evaluation
        working-directory: agent/klaudbiusz
        env:
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
        run: |
          uv run cli/evaluate_all.py \
            --mlflow-enabled \
            --experiment-name "/Shared/klaudbiusz-daily-evals" \
            --parallel 3 \
            --timeout 600

      - name: Publish apps
        working-directory: agent/klaudbiusz
        env:
          PUBLISH_REPO_PATH: ../../publish
        run: |
          uv run cli/publish_apps.py \
            --output-repo $PUBLISH_REPO_PATH \
            --push

      - name: Send alerts on failure
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          webhook-url: ${{ secrets.SLACK_WEBHOOK }}
          payload: |
            {
              "text": "Daily evaluation failed! Check workflow logs.",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "⚠️ *Daily Evaluation Failed*\n<${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}|View Logs>"
                  }
                }
              ]
            }

      - name: Upload evaluation report
        uses: actions/upload-artifact@v4
        with:
          name: evaluation-report-${{ github.run_number }}
          path: agent/klaudbiusz/app-eval/evaluation_report.json
          retention-days: 90

      - name: Cleanup Docker
        if: always()
        run: |
          docker system prune -af --volumes
```

#### 2.2 Databricks Workflow Alternative

For teams using Databricks, implement as a Databricks Workflow:

```python
# cli/databricks_daily_eval_job.py (NEW FILE)
"""
Databricks Job definition for daily evaluation.
Deploy with: databricks jobs create --json-file this_file.py
"""
import json

job_config = {
    "name": "klaudbiusz-daily-evaluation",
    "schedule": {
        "quartz_cron_expression": "0 0 2 * * ?",  # 2am daily
        "timezone_id": "UTC",
        "pause_status": "UNPAUSED"
    },
    "max_concurrent_runs": 1,
    "tasks": [
        {
            "task_key": "generate_apps",
            "description": "Generate 10 apps from daily prompts",
            "python_wheel_task": {
                "package_name": "klaudbiusz",
                "entry_point": "bulk_run",
                "parameters": [
                    "--prompts", "cli/prompts/daily_prompts.json",
                    "--max-apps", "10",
                    "--mode", "daily"
                ]
            },
            "libraries": [{"pypi": {"package": "klaudbiusz"}}],
            "new_cluster": {
                "spark_version": "14.3.x-scala2.12",
                "node_type_id": "i3.xlarge",
                "num_workers": 0,  # Single-node
                "spark_env_vars": {
                    "ANTHROPIC_API_KEY": "{{secrets/klaudbiusz/anthropic_api_key}}",
                    "DATABRICKS_HOST": "{{secrets/klaudbiusz/databricks_host}}",
                    "DATABRICKS_TOKEN": "{{secrets/klaudbiusz/databricks_token}}"
                }
            },
            "timeout_seconds": 7200  # 2 hours
        },
        {
            "task_key": "evaluate_apps",
            "description": "Run 9-metric evaluation",
            "depends_on": [{"task_key": "generate_apps"}],
            "python_wheel_task": {
                "package_name": "klaudbiusz",
                "entry_point": "evaluate_all",
                "parameters": [
                    "--mlflow-enabled",
                    "--experiment-name", "/Shared/klaudbiusz-daily-evals",
                    "--parallel", "5"
                ]
            },
            "libraries": [{"pypi": {"package": "klaudbiusz"}}],
            "new_cluster": {
                "spark_version": "14.3.x-scala2.12",
                "node_type_id": "i3.2xlarge",
                "num_workers": 0,
                "docker_image": {
                    "url": "ghcr.io/anthropic/klaudbiusz-eval:latest"
                }
            },
            "timeout_seconds": 10800  # 3 hours
        },
        {
            "task_key": "publish_results",
            "description": "Publish to GitHub and MLflow",
            "depends_on": [{"task_key": "evaluate_apps"}],
            "python_wheel_task": {
                "package_name": "klaudbiusz",
                "entry_point": "publish_apps",
                "parameters": ["--push"]
            },
            "libraries": [{"pypi": {"package": "klaudbiusz"}}],
            "existing_cluster_id": "{{cluster_id}}"
        }
    ],
    "email_notifications": {
        "on_failure": ["team@example.com"],
        "on_success": ["team@example.com"]
    },
    "webhook_notifications": {
        "on_failure": [
            {
                "id": "slack_webhook",
                "url": "{{secrets/klaudbiusz/slack_webhook_url}}"
            }
        ]
    }
}

if __name__ == "__main__":
    print(json.dumps(job_config, indent=2))
```

#### 2.3 Daily Prompts Rotation

Maintain a rotating set of prompts to test diverse scenarios:

```json
// cli/prompts/daily_prompts.json (NEW FILE)
{
  "version": "1.0",
  "prompts": [
    {
      "id": "customer-churn-prediction",
      "text": "Create a customer churn prediction dashboard with filters by segment and time period",
      "category": "ml-analytics",
      "complexity": "medium"
    },
    {
      "id": "sales-pipeline-tracker",
      "text": "Build a sales pipeline tracker showing deals by stage, rep, and expected close date",
      "category": "business-ops",
      "complexity": "simple"
    },
    {
      "id": "inventory-optimization",
      "text": "Create an inventory optimization dashboard with reorder suggestions and stock alerts",
      "category": "supply-chain",
      "complexity": "medium"
    },
    {
      "id": "financial-reporting",
      "text": "Build a monthly financial reporting app with P&L, balance sheet, and cash flow views",
      "category": "finance",
      "complexity": "high"
    },
    {
      "id": "user-engagement-metrics",
      "text": "Create a user engagement dashboard tracking DAU, retention, and feature adoption",
      "category": "product-analytics",
      "complexity": "medium"
    }
  ],
  "rotation_strategy": "round_robin",
  "daily_count": 5
}
```

#### 2.4 Quality Monitoring and Alerts

```python
# cli/quality_monitor.py (NEW FILE)
"""
Monitor daily evaluation results and send alerts on quality degradation.
"""
from datetime import date, timedelta
import json
from pathlib import Path

QUALITY_THRESHOLDS = {
    "avg_appeval_100": 80.0,  # Average quality score
    "success_rate": 0.85,      # 85% of apps should pass
    "regression_tolerance": 10.0  # Max points drop from previous day
}

def check_quality_regression(current_report: dict, previous_report: dict) -> list[dict]:
    """Compare current report against previous day."""
    issues = []

    current_avg = current_report['summary']['avg_appeval_100']
    previous_avg = previous_report['summary']['avg_appeval_100']

    # Check absolute threshold
    if current_avg < QUALITY_THRESHOLDS['avg_appeval_100']:
        issues.append({
            "severity": "high",
            "metric": "avg_appeval_100",
            "message": f"Quality below threshold: {current_avg:.1f} < {QUALITY_THRESHOLDS['avg_appeval_100']}",
            "current": current_avg,
            "threshold": QUALITY_THRESHOLDS['avg_appeval_100']
        })

    # Check regression
    regression = previous_avg - current_avg
    if regression > QUALITY_THRESHOLDS['regression_tolerance']:
        issues.append({
            "severity": "medium",
            "metric": "avg_appeval_100",
            "message": f"Quality regression detected: -{regression:.1f} points",
            "current": current_avg,
            "previous": previous_avg,
            "regression": regression
        })

    # Check success rate
    success_rate = current_report['summary']['success_rate']
    if success_rate < QUALITY_THRESHOLDS['success_rate']:
        issues.append({
            "severity": "high",
            "metric": "success_rate",
            "message": f"Success rate below threshold: {success_rate*100:.1f}% < {QUALITY_THRESHOLDS['success_rate']*100:.0f}%",
            "current": success_rate,
            "threshold": QUALITY_THRESHOLDS['success_rate']
        })

    return issues

def send_slack_alert(issues: list[dict]):
    """Send Slack notification with quality issues."""
    import requests
    import os

    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return

    severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "⚠️ Quality Alert - Daily Evaluation"}
        }
    ]

    for issue in issues:
        emoji = severity_emoji.get(issue['severity'], '⚪')
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{emoji} **{issue['metric']}**\n{issue['message']}"
            }
        })

    requests.post(webhook_url, json={"blocks": blocks})

def monitor_daily_evaluation():
    """Main monitoring function."""
    today = date.today()
    yesterday = today - timedelta(days=1)

    # Load reports
    eval_dir = Path(__file__).parent.parent / "app-eval"
    current_report_path = eval_dir / f"evaluation_report_{today}.json"
    previous_report_path = eval_dir / f"evaluation_report_{yesterday}.json"

    if not current_report_path.exists():
        print("❌ No evaluation report for today")
        return

    with open(current_report_path) as f:
        current_report = json.load(f)

    if previous_report_path.exists():
        with open(previous_report_path) as f:
            previous_report = json.load(f)

        # Check for regressions
        issues = check_quality_regression(current_report, previous_report)
    else:
        # First run, only check absolute thresholds
        issues = check_quality_regression(current_report, {"summary": {"avg_appeval_100": 100}})

    if issues:
        print(f"⚠️ Found {len(issues)} quality issues")
        for issue in issues:
            print(f"  {issue['severity'].upper()}: {issue['message']}")

        send_slack_alert(issues)
    else:
        print("✓ All quality checks passed")

if __name__ == "__main__":
    monitor_daily_evaluation()
```

---

### 3. Implementation Plan

**Phase 1: Publishing Infrastructure** (2-3 days)
- [ ] Create `anthropic/klaudbiusz-generated-apps` GitHub repository
- [ ] Implement `cli/publish_apps.py` script
- [ ] Test publishing flow with 5 sample apps
- [ ] Set up GitHub repository structure and README
- [ ] Configure GitHub Actions secrets

**Phase 2: Daily Evaluation Pipeline** (2 days)
- [ ] Create `cli/prompts/daily_prompts.json` with 20+ diverse prompts
- [ ] Implement GitHub Actions workflow (`.github/workflows/daily-eval.yml`)
- [ ] Test workflow with manual trigger
- [ ] Set up Slack webhooks for alerts
- [ ] Configure cron schedule for 2am UTC

**Phase 3: Quality Monitoring** (1-2 days)
- [ ] Implement `cli/quality_monitor.py`
- [ ] Define quality thresholds based on historical data
- [ ] Test alert system with simulated regressions
- [ ] Integrate monitoring into evaluation pipeline
- [ ] Document alert escalation process

**Phase 4: MLflow Artifacts** (1 day)
- [ ] Add `publish_app_artifacts()` to `mlflow_tracker.py`
- [ ] Test artifact upload for full app sources
- [ ] Configure MLflow artifact retention policies
- [ ] Document artifact access for team

**Phase 5: Documentation and Launch** (1 day)
- [ ] Create public dashboard (GitHub Pages or Streamlit)
- [ ] Write reproduction guide (`docs/REPRODUCTION.md`)
- [ ] Document metrics (`docs/METRICS.md`)
- [ ] Announce to team and community
- [ ] Monitor first week of daily runs

---

### 4. Cost Estimation

**Daily Pipeline Costs:**

| Resource | Usage | Cost/Day | Cost/Month |
|----------|-------|----------|------------|
| GitHub Actions (3h @ 8 cores) | 24 core-hours | $1.92 | $57.60 |
| Claude API (10 apps × 50K tokens avg) | 500K tokens | $7.50 | $225.00 |
| Databricks Warehouse (SQL eval) | 2 DBU-hours | $1.00 | $30.00 |
| MLflow Storage (10 apps × 50MB) | 500MB/day | $0.01 | $0.30 |
| Docker Hub Bandwidth | 5GB/day | $0.00 | $0.00 |
| **Total** | | **~$10.43** | **~$313** |

**Cost Optimization:**
- Use smaller model (Haiku) for simple apps: -50% API cost
- Cache Docker images: -30% build time
- Run on Databricks clusters: -100% GitHub Actions cost (if already have cluster)

**Budget Recommendation:** $400/month with 25% buffer

---

### 5. Success Metrics

Track these to validate the publishing and daily eval system:

1. **Pipeline Reliability**
   - Metric: % of successful daily runs
   - Target: >95% success rate

2. **Time to Publish**
   - Metric: Minutes from eval completion to GitHub publish
   - Target: <10 minutes

3. **Community Engagement** (if public)
   - Metric: GitHub stars, issues filed, PRs submitted
   - Target: 10+ stars in first month

4. **Quality Trend**
   - Metric: 30-day moving average of appeval_100
   - Target: Stable or improving (no >5pt drop)

5. **Alert Accuracy**
   - Metric: % of alerts that identify real regressions
   - Target: >80% true positive rate

---

### 6. Public Dashboard

Create a live dashboard showing daily evaluation results:

**Option A: GitHub Pages + Chart.js**
- Static site generated from `metrics_timeseries.csv`
- Updates via GitHub Actions commit
- Free hosting

**Option B: Streamlit + MLflow**
- Dynamic dashboard pulling from MLflow
- Hosted on Databricks or Streamlit Cloud
- More interactive, real-time data

```python
# dashboard/streamlit_app.py (NEW FILE)
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Klaudbiusz Daily Evaluations", layout="wide")

st.title("Klaudbiusz: AI-Generated Databricks Apps")
st.markdown("**Daily quality monitoring of 90%+ success rate app generation**")

# Load metrics timeseries
metrics_path = Path("evaluations/metrics_timeseries.csv")
df = pd.read_csv(metrics_path)
df['date'] = pd.to_datetime(df['date'])

# Display current stats
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Avg Quality Score", f"{df['avg_appeval_100'].iloc[-1]:.1f}/100")
with col2:
    st.metric("Success Rate", f"{df['success_rate'].iloc[-1]*100:.0f}%")
with col3:
    st.metric("Total Apps Generated", len(df) * 10)
with col4:
    st.metric("Days Running", len(df))

# Plot trends
st.subheader("Quality Trends")
st.line_chart(df.set_index('date')[['avg_appeval_100']])

st.subheader("Recent Apps")
# Load latest evaluation report
import json
with open("evaluations/2025-01-20_report.json") as f:
    report = json.load(f)

for app in report['apps'][:10]:
    with st.expander(f"{app['app_name']} - {app['metrics']['appeval_100']:.0f}/100"):
        st.markdown(f"**Prompt:** {app.get('prompt', 'N/A')}")
        st.markdown(f"**Build:** {'✓' if app['metrics']['build_success'] else '✗'}")
        st.markdown(f"**Type Safety:** {'✓' if app['metrics']['type_safety'] else '✗'}")
        st.markdown(f"[View Source](./apps/2025-01-20/{app['app_name']})")
```

---

## References

- **Databricks Docs:** https://docs.databricks.com/aws/en/mlflow3/genai/eval-monitor/
- **MLflow Tracing TypeScript SDK:** https://docs.databricks.com/aws/en/mlflow3/genai/tracing/app-instrumentation/typescript-sdk
- **MLflow Custom Metrics:** https://mlflow.org/docs/latest/llms/llm-evaluate/index.html
- **Current Implementation:**
  - `cli/evaluate_all.py` - Batch evaluation
  - `cli/eval_metrics.py` - Metric calculations
  - `cli/mlflow_tracker.py` - MLflow logging
  - `eval-docs/evals.md` - Metrics documentation
- **Proposed Implementation:**
  - `edda_templates/template_trpc/server/src/index.ts` - Instrumented template
  - `cli/mlflow_runtime_scorers.py` - Runtime scorers
  - `cli/setup_production_monitoring.py` - Monitoring setup
  - `cli/improvement_loop.py` - Meta-agentic feedback loop

---

**Document Status:** Draft
**Last Updated:** 2025-01-19
**Owner:** Klaudbiusz Team
**Reviewers:** TBD
