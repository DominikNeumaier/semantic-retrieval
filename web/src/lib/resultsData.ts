/**
 * Aggregators for the results dashboard, reading directly from the restructured
 * trace tree under ../results/. Two top-level axes:
 *
 *   results/orchestration/  — natural end-to-end runs (routing, gap-finding,
 *                             decomposition). Method A / clean is the reference.
 *   results/retrieval/      — isolated Top-1/Top-5 per condition (method x state),
 *                             plus the design-time evaluation.
 *
 * Everything the page needs is recomputed here from the traces, so the dashboard
 * is the single consumer of the tree and no pre-aggregated summary files are read
 * (apart from the design-time tables, which are reused verbatim).
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// src/lib/ → web/ → semantic-retrieval/  →  ROOT = semantic-retrieval/
const ROOT = path.resolve(__dirname, "..", "..", "..");

export const ORCH = path.join(ROOT, "results", "orchestration");
export const RET = path.join(ROOT, "results", "retrieval");

export const METHODS = ["A", "B", "C", "D", "E", "S", "F"];
export const STATES = ["0", "1"];
export const CONDITIONS = METHODS.flatMap(m => STATES.map(s => m + s));

export const METHOD_DESC: Record<string, string> = {
  A: "Embedding",
  B: "Progressive",
  C: "Graph",
  D: "Agentic Tools",
  E: "Agentic Raw",
  S: "Baseline",
  F: "Agentic Hybrid",
};

// Agentic methods where a termination breakdown (picked / refused / budget) is
// meaningful — the others always return a ranked list.
const AGENTIC = new Set(["D", "E", "F"]);

// ─── helpers ────────────────────────────────────────────────────────────────

export function loadJson(p: string): any {
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : [];
}

export function loadText(p: string): string {
  return fs.existsSync(p) ? fs.readFileSync(p, "utf8") : "";
}

export const methodOf = (cond: string) => cond[0];
export const stateOf = (cond: string) => (cond[1] === "1" ? "enriched" : "clean");
export const ordLabel = (id: string) => (id ? id.split(":")[2] || id : id);
export const ordSlug = (id: string) => (id || "").replace(/[:.]/g, "-");

export const mean = (xs: number[]) =>
  xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
export const round3 = (x: number) => Math.round(x * 1000) / 1000;

/** Wilson 95% score interval for k successes of n trials. */
export function wilson95(k: number, n: number): { lo: number; hi: number } {
  if (n === 0) return { lo: 0, hi: 0 };
  const z = 1.959964;
  const p = k / n;
  const z2 = z * z;
  const denom = 1 + z2 / n;
  const center = p + z2 / (2 * n);
  const half = z * Math.sqrt((p * (1 - p) + z2 / (4 * n)) / n);
  return { lo: (center - half) / denom, hi: (center + half) / denom };
}

// ─── memoized trace loaders ───────────────────────────────────────────────────

const _dirCache = new Map<string, any[]>();

/** Parse every *.json in absDir, sorted by filename. Memoized per directory. */
export function loadTraceDir(absDir: string): any[] {
  if (_dirCache.has(absDir)) return _dirCache.get(absDir)!;
  let out: any[] = [];
  if (fs.existsSync(absDir) && fs.statSync(absDir).isDirectory()) {
    out = fs
      .readdirSync(absDir)
      .filter(f => f.endsWith(".json"))
      .sort()
      .map(f => {
        try {
          return JSON.parse(fs.readFileSync(path.join(absDir, f), "utf8"));
        } catch {
          return null;
        }
      })
      .filter(Boolean);
  }
  _dirCache.set(absDir, out);
  return out;
}

/** { COND: traces } for every method x state under baseDir. Memoized via loadTraceDir. */
export function loadCondTree(baseDir: string): Record<string, any[]> {
  const tree: Record<string, any[]> = {};
  for (const cond of CONDITIONS) {
    const traces = loadTraceDir(path.join(baseDir, cond));
    if (traces.length) tree[cond] = traces;
  }
  return tree;
}

// ─── Orchestration summary (Tab A) ────────────────────────────────────────────

export function buildOrchestrationSummary() {
  // Skill-Guided: end-to-end orchestration on the 30 canonical benchmark prompts.
  // The pipeline runs with no mode hint, so it must both detect the mode
  // (predicted_mode == skill_guided) AND pick the right skill. Regenerated
  // clean on the current cases (30 distinct prompts, no duplicates). Routing is
  // method/state-independent for SG (retrieval is bypassed), so A1 is the
  // representative reference condition.
  const SG_REF = "A1";
  const sgTraces = loadTraceDir(path.join(ORCH, "skill_guided", "traces", SG_REF));
  const sgCases = sgTraces.map((r: any) => {
    const ot = r.orchestration_trace || {};
    const it = ot.intent_trace || ot.intent_resolver || {};
    return {
      case_id: r.case_id,
      user_prompt: r.user_prompt,
      mode_expected: r.mode_expected,
      predicted_mode: r.metrics?.predicted_mode ?? null,
      mode_routing_ok: r.metrics?.mode_routing_ok ?? 0,
      skill_routing_ok: r.skill_picked === r.skill_expected ? 1 : 0,
      coverage: r.plan?.coverage ?? ot.planner_trace?.coverage ?? null,
      skill_expected: r.skill_expected,
      skill_picked: r.skill_picked,
      reason: ot.planner_trace?.reason || it.reason || "",
    };
  });
  const sg = {
    cases: sgCases.length,
    routingAcc: round3(mean(sgTraces.map((r: any) => r.metrics?.mode_routing_ok || 0))),
    skillAcc: round3(mean(sgCases.map((c: any) => c.skill_routing_ok))),
    tokensPerCall: sgTraces.length
      ? Math.round(mean(sgTraces.map((r: any) => r.trace?.tokens || 0)))
      : 0,
    ref: SG_REF,
  };

  // Skill-Adjusted: natural runs, reference condition A0 (traces/ subdir).
  const saTraces = loadTraceDir(path.join(ORCH, "skill_adjusted", "traces", "A0"));
  const gapSources = new Set(["gap", "gap_forced"]);
  const saRouting = saTraces.map((r: any) => {
    const steps = (r.plan?.steps || []).filter((s: any) => gapSources.has(s.source));
    const predictedCount = steps.length;
    const expectedCount = (r.expected_gaps || []).length;
    return {
      case_id: r.case_id,
      user_prompt: r.user_prompt,
      skill_expected: r.skill_expected,
      skill_picked: r.skill_picked,
      skill_routing_ok: r.skill_picked === r.skill_expected ? 1 : 0,
      expected_gaps: r.expected_gaps || [],
      expected_gap_count: expectedCount,
      predicted_gap_steps: steps.map((s: any) => s.step_name),
      predicted_gap_count: predictedCount,
      gap_detection: r.metrics?.gap_detection ?? null,
      routing_ok: r.metrics?.mode_routing_ok ?? 0,
    };
  });
  const sa = {
    cases: saTraces.length,
    routingAcc: round3(mean(saTraces.map((r: any) => r.metrics?.mode_routing_ok || 0))),
    skillAcc: round3(mean(saRouting.map((c: any) => c.skill_routing_ok))),
    gapFoundRate: round3(
      mean(saTraces.map((r: any) =>
        (r.plan?.steps || []).some((s: any) => gapSources.has(s.source)) ? 1 : 0))
    ),
    gapCountAcc: round3(
      mean(saRouting.map(c => (c.predicted_gap_count === c.expected_gap_count ? 1 : 0)))
    ),
    tokensPerCase: saTraces.length
      ? Math.round(mean(saTraces.map((r: any) => r.trace?.tokens || 0)))
      : 0,
  };

  // Dynamic: natural runs, reference condition A0 (routing + decomposition).
  const dyTraces = loadTraceDir(path.join(ORCH, "dynamic", "A0"));
  const decompByCase: Record<string, any> = {};
  let green = 0, yellow = 0, red = 0, decompTotal = 0;
  const dyRouting = dyTraces.map((r: any) => {
    const decomp = r.orchestration_trace?.decomposer_routing ?? null;
    if (decomp) {
      decompByCase[r.case_id] = decomp;
      decompTotal++;
      if (decomp.verdict === "green") green++;
      else if (decomp.verdict === "yellow") yellow++;
      else if (decomp.verdict === "red") red++;
    }
    return {
      case_id: r.case_id,
      user_prompt: r.user_prompt,
      routing_ok: r.metrics?.mode_routing_ok ?? 0,
      predicted_mode: r.metrics?.predicted_mode ?? null,
      skill_picked: r.skill_picked,
      reason: r.orchestration_trace?.intent_resolver?.reason ?? "",
      decomp,
    };
  });
  const dy = {
    cases: dyTraces.length,
    routingAcc: round3(mean(dyTraces.map((r: any) => r.metrics?.mode_routing_ok || 0))),
    decompAcc: decompTotal ? round3(green / decompTotal) : null,
    decompTally: { green, yellow, red, total: decompTotal },
    tokensPerCase: dyTraces.length
      ? Math.round(mean(dyTraces.map((r: any) => r.trace?.tokens || 0)))
      : 0,
  };

  return { sg, sa, dy, sgCases, saRouting, dyRouting, decompByCase };
}

// ─── Retrieval: Design-Time (Tab B) ───────────────────────────────────────────

export function buildRetrievalDesignTime() {
  const summary = loadJson(path.join(RET, "design-time", "summary.json"));
  const byProcess = loadJson(path.join(RET, "design-time", "by_process.json"));
  const byNotation = loadJson(path.join(RET, "design-time", "by_modeling_notation.json"));
  return { summary, byProcess, byNotation };
}

/** Per-process design-time step traces for one method. */
export function loadDesignTimeTraces(method: string): any[] {
  return loadTraceDir(path.join(RET, "design-time", "traces", method));
}

// ─── Retrieval: Skill-Guided (skill selection) ────────────────────────────────

/**
 * Skill-selection evaluation: given a skill-guided prompt, did the resolver pick
 * the correct skill id? Method-independent (the resolver ranks over the skill
 * library, not the retrieval backend), so the traces are a single flat set.
 */
export function buildRetrievalSkillGuided() {
  const dir = path.join(RET, "runtime", "skill_guided", "traces");
  const summaryFile = loadJson(path.join(dir, "summary.json"));
  const files = loadTraceDir(dir).filter((r: any) => r && r.case_id);
  const cases = files.map((r: any) => ({
    case_id: r.case_id,
    user_prompt: r.user_prompt,
    skill_expected: r.skill_expected,
    skill_picked: r.skill_picked,
    routing_ok: r.routing_ok,
    reason: r.reason,
  }));
  const correct = cases.filter((c: any) => c.routing_ok === 1).length;
  const tokens = files.reduce((a: number, r: any) => a + (r.tokens || 0), 0);
  return {
    summary: {
      cases: summaryFile.cases ?? cases.length,
      routingAcc: summaryFile["Routing-Acc"] ?? (cases.length ? round3(correct / cases.length) : 0),
      correct: summaryFile.correct ?? correct,
      tokensPerCase: cases.length ? Math.round(tokens / cases.length) : 0,
    },
    cases,
  };
}

// ─── Retrieval: Skill-Adjusted (Tab B) ────────────────────────────────────────

export function buildRetrievalSA() {
  const gapSources = new Set(["gap", "gap_forced"]);
  const rows: any[] = [];
  const tracesByCond: Record<string, any[]> = {};

  for (const cond of CONDITIONS) {
    const traces = loadTraceDir(path.join(RET, "runtime", "skill_adjusted", "traces", cond));
    if (!traces.length) continue;
    const n = traces.length;
    let top1Sum = 0, top5Sum = 0, top1Found = 0, top5Found = 0, nGap = 0;
    const tokens: number[] = [];
    const entries: any[] = [];

    for (const r of traces) {
      const expected: string[] = r.expected_gaps || [];
      const expectedSet = new Set(expected);
      const steps = (r.plan?.steps || []).filter((s: any) => gapSources.has(s.source));
      const wasForced = (r.condition || cond).length > 2; // A0gf / F0f carry a suffix
      tokens.push(r.trace?.tokens || 0);

      // fraction of expected gap ordIds hit at rank-1 / within top-5 by any gap step
      let t1 = 0, t5 = 0;
      const stepInfo: any[] = [];
      for (const step of steps) {
        const cands = (step.candidates || []).map((c: any) => c.ordId);
        stepInfo.push({
          name: (step.step_name || "").slice(0, 60),
          top1: cands[0] ?? "none",
          top5: cands.slice(0, 5),
        });
      }
      if (expectedSet.size > 0) {
        const hit1 = new Set<string>();
        const hit5 = new Set<string>();
        for (const step of steps) {
          const cands = (step.candidates || []).map((c: any) => c.ordId);
          if (expectedSet.has(cands[0])) hit1.add(cands[0]);
          for (const id of cands.slice(0, 5)) if (expectedSet.has(id)) hit5.add(id);
        }
        t1 = hit1.size / expectedSet.size;
        t5 = hit5.size / expectedSet.size;
      }
      top1Sum += t1;
      top5Sum += t5;
      if (steps.length > 0) {
        nGap++;
        top1Found += t1;
        top5Found += t5;
      }
      entries.push({
        case_id: r.case_id,
        user_prompt: r.user_prompt,
        skill: r.skill_picked,
        expected_gaps: expected,
        gap_detection: r.metrics?.gap_detection ?? null,
        routing_ok: r.metrics?.mode_routing_ok ?? null,
        was_forced: wasForced,
        top1_correct: t1,
        top5_correct: t5,
        gap_steps: stepInfo,
      });
    }

    rows.push({
      cond,
      method: methodOf(cond),
      state: stateOf(cond),
      n,
      top1: round3(top1Sum / n),
      top5: round3(top5Sum / n),
      nGap,
      top1Gap: nGap ? round3(top1Found / nGap) : null,
      top5Gap: nGap ? round3(top5Found / nGap) : null,
      tokensPerCase: Math.round(mean(tokens)),
    });
    tracesByCond[cond] = entries;
  }

  rows.sort((a, b) => a.cond.localeCompare(b.cond));
  return { rows, tracesByCond };
}

// ─── Retrieval: Dynamic (Tab B) ───────────────────────────────────────────────

function terminationOf(methodTrace: any, refusedFlag: boolean): "picked" | "refused" | "budget_exhausted" {
  if (refusedFlag || methodTrace?.refused === true) return "refused";
  if (methodTrace?.picked_ord_id != null) return "picked";
  return "budget_exhausted";
}

export function buildRetrievalDynamic() {
  // case_id → natural user_prompt for multi cases (only sub_query lives on the subs)
  const promptByCase: Record<string, string> = {};
  for (const r of loadTraceDir(path.join(ORCH, "dynamic", "A0"))) {
    promptByCase[r.case_id] = r.user_prompt;
  }

  const rows: any[] = [];
  const tracesByCond: Record<string, any[]> = {};
  const termination: Record<string, any> = {};

  for (const cond of CONDITIONS) {
    const all = loadTraceDir(path.join(RET, "runtime", "dynamic", "traces", cond));
    if (!all.length) continue;
    // singles carry a metrics object and no sub_idx; subs carry sub_idx
    const singleTraces = all.filter((r: any) => !("sub_idx" in r) && r.metrics);
    const subTraces = all.filter((r: any) => "sub_idx" in r);

    // singles: nested metrics
    const singleTop1 = singleTraces.map((r: any) => (r.metrics?.top1_acc ? 1 : 0));
    const singleTop5 = singleTraces.map((r: any) => (r.metrics?.candidate_recall ? 1 : 0));
    const nSingle = singleTraces.length;

    // subs grouped by case
    const byCase: Record<string, any[]> = {};
    for (const s of subTraces) (byCase[s.case_id] ||= []).push(s);
    for (const cid in byCase) byCase[cid].sort((a, b) => (a.sub_idx || 0) - (b.sub_idx || 0));
    const multiCases = Object.keys(byCase).sort();
    const nMulti = multiCases.length;

    const multiCredit1 = multiCases.map(cid =>
      mean(byCase[cid].map(s => (s.top1_acc ? 1 : 0))));
    const multiCredit5 = multiCases.map(cid =>
      mean(byCase[cid].map(s => (s.candidate_recall ? 1 : 0))));
    const recallAllMulti = nMulti
      ? mean(multiCases.map(cid => (byCase[cid].every(s => s.candidate_recall === 1) ? 1 : 0)))
      : null;

    const nTotal = nSingle + nMulti;
    const sumSingle1 = singleTop1.reduce((a, b) => a + b, 0);
    const sumSingle5 = singleTop5.reduce((a, b) => a + b, 0);
    const sumMulti1 = multiCredit1.reduce((a, b) => a + b, 0);
    const sumMulti5 = multiCredit5.reduce((a, b) => a + b, 0);
    const combinedTop1 = nTotal ? (sumSingle1 + sumMulti1) / nTotal : 0;
    const combinedTop5 = nTotal ? (sumSingle5 + sumMulti5) / nTotal : 0;

    // Wilson CI on the blended value (approx: k=round(Σcredit), n=nTotal)
    const ci = wilson95(Math.round(sumSingle1 + sumMulti1), nTotal);

    // tokens: singles trace.tokens + subs tokens, per case
    const singleTokens = singleTraces.reduce((a: number, r: any) => a + (r.trace?.tokens || 0), 0);
    const subTokens = subTraces.reduce((a: number, r: any) => a + (r.tokens || 0), 0);
    const tokensPerCase = nTotal ? Math.round((singleTokens + subTokens) / nTotal) : 0;

    // Termination (agentic methods only), over every retrieval unit
    let term: any = null;
    if (AGENTIC.has(methodOf(cond))) {
      const counts = { picked: 0, refused: 0, budget_exhausted: 0, total: 0 };
      for (const r of singleTraces) {
        const mt = r.plan?.steps?.[0]?.method_trace;
        counts[terminationOf(mt, false)]++;
        counts.total++;
      }
      for (const s of subTraces) {
        const mt = s.plan?.steps?.[0]?.method_trace;
        counts[terminationOf(mt, s.refused === true)]++;
        counts.total++;
      }
      term = counts;
    }
    termination[cond] = term;

    rows.push({
      cond,
      method: methodOf(cond),
      state: stateOf(cond),
      n: nTotal,
      top1: round3(combinedTop1),
      top1CIlo: round3(ci.lo),
      top1CIhi: round3(ci.hi),
      top5: round3(combinedTop5),
      top1Single: nSingle ? round3(sumSingle1 / nSingle) : null,
      top1Multi: nMulti ? round3(sumMulti1 / nMulti) : null,
      recallAllMulti: recallAllMulti != null ? round3(recallAllMulti) : null,
      tokensPerCase,
    });

    // viewer entries
    const entries: any[] = [];
    for (const r of singleTraces) {
      const step = r.plan?.steps?.[0];
      entries.push({
        case_id: r.case_id,
        user_prompt: r.user_prompt,
        query_class: "single_intent",
        top1_correct: r.metrics?.top1_acc === 1,
        top5_correct: r.metrics?.candidate_recall === 1,
        expected_ordIds: step?.expected?.expected_ordIds || [],
        top5_candidates: (step?.candidates || []).slice(0, 5).map((c: any) => c.ordId),
        method_trace: AGENTIC.has(methodOf(cond)) ? step?.method_trace ?? null : null,
      });
    }
    for (const cid of multiCases) {
      const subs = byCase[cid];
      entries.push({
        case_id: cid,
        user_prompt: promptByCase[cid] || "",
        query_class: "multi_intent",
        top1_correct: subs.every(s => s.top1_acc === 1),
        top5_correct: subs.some(s => s.candidate_recall === 1),
        expected_ordIds: subs.map(s => s.gt_id),
        subs: subs.map(s => ({
          sub_query: s.sub_query,
          gt_id: s.gt_id,
          top1_acc: s.top1_acc,
          candidate_recall: s.candidate_recall,
          refused: s.refused === true,
          top5_candidates: (s.top5_candidates || []).slice(0, 5),
        })),
      });
    }
    entries.sort((a, b) => a.case_id.localeCompare(b.case_id));
    tracesByCond[cond] = entries;
  }

  rows.sort((a, b) => a.cond.localeCompare(b.cond));
  return { rows, tracesByCond, termination };
}

// ─── Retrieval: Out-of-Scope (Tab B) ──────────────────────────────────────────

export function buildRetrievalOOS() {
  const rows: any[] = [];
  const tracesByCond: Record<string, any[]> = {};

  for (const cond of CONDITIONS) {
    const traces = loadTraceDir(path.join(RET, "runtime", "out_of_scope", "traces", cond));
    if (!traces.length) continue;
    const n = traces.length;
    rows.push({
      cond,
      method: methodOf(cond),
      state: stateOf(cond),
      n,
      refusalRate: round3(mean(traces.map((r: any) => r.metrics?.correctly_refused || 0))),
      falsePickRate: round3(mean(traces.map((r: any) => r.metrics?.falsely_picked || 0))),
      tokensPerCase: Math.round(mean(traces.map((r: any) => r.trace?.tokens || 0))),
    });
    tracesByCond[cond] = traces.map((r: any) => ({
      case_id: r.case_id,
      user_prompt: r.user_prompt,
      topic: r.topic ?? null,
      correctly_refused: r.metrics?.correctly_refused === 1,
      falsely_picked_ord_id: r.metrics?.falsely_picked_ord_id ?? null,
      picked_steps: ((r.plan?.steps ?? []) as any[])
        .filter((s) => Array.isArray(s.candidates) && s.candidates.length)
        .map((s) => ({
          step_name: s.step_name,
          ord_id: s.method_trace?.picked_ord_id || s.candidates[0]?.ordId || null,
        })),
    }));
  }

  rows.sort((a, b) => a.cond.localeCompare(b.cond));
  return { rows, tracesByCond };
}

// ─── Status strip ─────────────────────────────────────────────────────────────

export function buildStatus() {
  const count = (p: string) => loadTraceDir(p).length;
  return {
    sg: count(path.join(ORCH, "skill_guided", "traces", "A1")),
    saOrch: count(path.join(ORCH, "skill_adjusted", "traces", "A0")),
    dyOrch: count(path.join(ORCH, "dynamic", "A0")),
    designTime: count(path.join(RET, "design-time", "traces", "A")),
    retSA: CONDITIONS.reduce((a, c) => a + count(path.join(RET, "runtime", "skill_adjusted", "traces", c)), 0),
    retDY: CONDITIONS.reduce((a, c) => a + count(path.join(RET, "runtime", "dynamic", "traces", c)), 0),
    retOOS: CONDITIONS.reduce((a, c) => a + count(path.join(RET, "runtime", "out_of_scope", "traces", c)), 0),
  };
}
