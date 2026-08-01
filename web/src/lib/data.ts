/**
 * Data loaders for semantic-retrieval results browser.
 * Reads evaluation results from ../results/ and test cases from ../data/test_cases/
 * at build time.
 *
 * Path layout (relative to the web/ folder):
 *   ../results/design-time/    — DT evaluation summaries and traces
 *   ../results/runtime/        — RT evaluation records, summaries, routing
 *   ../data/test_cases/        — ground-truth case definitions
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// src/lib/ → web/ → semantic-retrieval/  →  ROOT = semantic-retrieval/
const ROOT = path.resolve(__dirname, "..", "..", "..");

export function loadJson(p: string): any {
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : [];
}

export function loadText(p: string): string {
  return fs.existsSync(p) ? fs.readFileSync(p, "utf8") : "";
}

// ─── Runtime test case types (for SA gap counts) ─────────────────────────────

export type RuntimeCase = {
  case_id: string;
  mode: "skill_guided" | "skill_adjusted" | "dynamic" | "out_of_scope";
  query_class: string;
  user_prompt: string;
  expected_skill_id?: string;
  expected_ordIds?: string[];
  expected_gap_ordIds?: string[];
};

export function loadRuntimeCasesForMode(mode: string): RuntimeCase[] {
  const p = path.join(ROOT, "data", "test_cases", "runtime", "output", `${mode}.json`);
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : [];
}

// ─── Results paths ────────────────────────────────────────────────────────────

export const RESULTS_ROOT = path.join(ROOT, "results");

export function loadRecords(mode: string): any[] {
  const p = path.join(RESULTS_ROOT, "runtime", mode, "records.jsonl");
  if (!fs.existsSync(p)) return [];
  const lines = fs.readFileSync(p, "utf8").split("\n").filter(l => l.trim());
  return lines.flatMap(l => { try { return [JSON.parse(l)]; } catch { return []; } });
}

export function loadMhRecords(): any[] {
  const p = path.join(RESULTS_ROOT, "runtime", "dynamic", "records_mh.jsonl");
  if (!fs.existsSync(p)) return [];
  const out: any[] = [];
  for (const l of fs.readFileSync(p, "utf8").split("\n")) {
    if (!l.trim()) continue;
    try { out.push(JSON.parse(l)); } catch {}
  }
  return out;
}

export function countRecords(mode: string): number {
  const p = path.join(RESULTS_ROOT, "runtime", mode, "records.jsonl");
  if (!fs.existsSync(p)) return 0;
  const lines = fs.readFileSync(p, "utf8").split("\n").filter(l => l.trim());
  return lines.filter(l => {
    try {
      const r = JSON.parse(l);
      const cond = r.condition ?? '';
      if (cond.startsWith('F') && cond.endsWith('f')) return true;
      return !cond.endsWith('f');
    } catch { return false; }
  }).length;
}
