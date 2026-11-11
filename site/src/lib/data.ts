// Build-time data access. Everything on the site comes from ../results and ../project.toml.
import { existsSync, readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parse } from "smol-toml";

const root = pathToFileURL(resolve(process.cwd(), "..") + "/");   // the site builds from site/
export const project = (parse(readFileSync(new URL("project.toml", root), "utf8")) as any).project;
// Set SNAPME_RESULTS_DIR to build from a results folder containing snapme_v1/ and nutrition5k_v1/.
const resultsRoot = process.env.SNAPME_RESULTS_DIR
  ? pathToFileURL(resolve(process.env.SNAPME_RESULTS_DIR) + "/") : new URL("results/", root);
const results = new URL("snapme_v1/", resultsRoot);

export type Row = Record<string, any>;

function json<T>(name: string, fallback: T): T {
  const p = new URL(name, results);
  return existsSync(p) ? (JSON.parse(readFileSync(p, "utf8")) as T) : fallback;
}

export const allRuns = (): Row[] => json<Row[]>("leaderboard.json", []);
export const runs = (pipeline: string): Row[] => allRuns().filter((r) => r.pipeline === pipeline);
export const baselines = (): Row[] => json<Row[]>("baselines_2023.json", []);
export const notScored = (): Row[] => json<Row[]>("not_scored.json", []);
export const gtBuild = (): any => json<any>("gt_build.json", null);
export const retrieval = (): any => json<any>("retrieval_embeddinggemma2.json", null);
export const mappingCheck = (): any => json<any>("mapping_check.json", null);
export const markerAblation = (): any => json<any>("marker_ablation.json", null);
export const beforeAfter = (): Row[] => json<Row[]>("before_after.json", []);
export function nutrition5k(): Row[] {
  const p = new URL("../nutrition5k_v1/leaderboard.json", results);
  return existsSync(p) ? JSON.parse(readFileSync(p, "utf8")) : [];
}

export function builtOn(): string {
  const p = new URL("leaderboard.json", results);
  if (!existsSync(p)) return "not yet";
  return statSync(p).mtime.toISOString().slice(0, 10);
}

export const base = import.meta.env.BASE_URL.replace(/\/$/, "");
export const href = (path: string) => `${base}/${path.replace(/^\//, "")}`;
export const repoUrl = (path = "") => `https://github.com/${project.github_owner}/${project.slug}${path ? `/blob/main/${path}` : ""}`;

export const fmt = {
  f: (x: number | null | undefined, d = 2) => (x == null ? "–" : x.toFixed(d)),
  pct: (x: number | null | undefined, d = 0) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`),
  n: (x: number | null | undefined, d = 0) =>
    x == null ? "–" : x.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }),
  ci: (c: number[] | null | undefined, d = 2) => (c ? `${c[0].toFixed(d)}–${c[1].toFixed(d)}` : ""),
  ciPct: (c: number[] | null | undefined) => (c ? `${(c[0] * 100).toFixed(0)}–${(c[1] * 100).toFixed(0)}%` : ""),
};

/** Model name as its developer publishes it (the Hugging Face repository name). */
export const modelName = (r: Row): string => (r.hf_repo ? String(r.hf_repo).split("/")[1] : r.model);

export const params = (r: Row): string =>
  r.params_total_b == null ? "–" :
  r.params_active_b == null && /mixture[ -]+of[ -]+experts|\bmoe\b/i.test(r.architecture ?? "") ?
    `${r.params_total_b} (active count unavailable)` :
  r.params_active_b && r.params_active_b !== r.params_total_b ? `${r.params_total_b} (${r.params_active_b} active)` :
  `${r.params_total_b}`;
