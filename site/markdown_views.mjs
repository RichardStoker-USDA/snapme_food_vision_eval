// After building the HTML, derive a Markdown copy of each page (<page>/index.md) and an llms.txt index from it.
// This keeps the Markdown in step with the page. Omit controls, scripts and dark-theme figure copies;
// Keep intervals in parentheses and separate detail text with middle dots.
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { NodeHtmlMarkdown } from "node-html-markdown";

const nhm = new NodeHtmlMarkdown({ bulletMarker: "-", maxConsecutiveNewlines: 2, useLinkReferenceDefinitions: false });

function clean(html, origin) {
  return html
    // Replace embedded figures with image links to their light-theme SVG files.
    .replace(/<figure([^>]*)data-md-src="([^"]+)"([^>]*)data-md-alt="([^"]*)"([^>]*)>/gi, '<figure$1$3$5><img src="$2" alt="$4">')
    .replace(/<div class="fig-(light|dark)"[^>]*>[\s\S]*?<\/svg>\s*<\/div>/gi, "")
    .replace(/<(script|style|dialog|form|svg|template)\b[\s\S]*?<\/\1>/gi, "")
    .replace(/<img[^>]*class="fig-dark"[^>]*>/gi, "")
    .replace(/<(button|select|input|iframe)\b[^>]*>(?:[\s\S]*?<\/\1>)?/gi, (m, tag) =>
      tag === "button" ? m.replace(/<\/?button[^>]*>/gi, "") : "")
    .replace(/<span class="ci"[^>]*>([\s\S]*?)<\/span>/gi, (_, t) => (t.trim() ? ` (${t.trim()})` : ""))
    .replace(/<span class="sub"[^>]*>([\s\S]*?)<\/span>/gi, (_, t) => (t.trim() ? ` · ${t.trim()}` : ""))
    .replace(/(href|src)="\/(?!\/)/g, `$1="${origin}/`);
}

export default function markdownViews({ site, base, name, tagline }) {
  return {
    name: "markdown-views",
    hooks: {
      "astro:build:done": ({ dir, pages }) => {
        const root = fileURLToPath(dir);
        const prefix = `${site}${base}`;
        const entries = [];
        for (const { pathname } of pages) {
          const file = join(root, pathname, "index.html");
          if (!existsSync(file)) continue;
          const html = readFileSync(file, "utf8");
          const main = html.match(/<main[^>]*>([\s\S]*?)<\/main>/i)?.[1];
          if (!main) continue;
          const title = (html.match(/<title>([^<]*)<\/title>/i)?.[1] ?? name).replace(/&amp;/g, "&");
          const desc = html.match(/<meta name="description" content="([^"]*)"/i)?.[1]?.replace(/&amp;/g, "&") ?? "";
          const url = `${prefix}/${pathname}`;
          // Restore readable punctuation: %5F in URLs and "\." except at numbered-list line starts.
          const md = nhm.translate(clean(main, site)).trim().replace(/%5F/g, "_").split("\n")
            .map((l) => (/^\s*\d+\\\./.test(l) ? l : l.replace(/\\\./g, "."))).join("\n");
          writeFileSync(join(root, pathname, "index.md"), `${md}\n\n---\nSource page: ${url}\n`);
          entries.push({ title: pathname ? title.split(" · ")[0] : "Results", url, desc });
        }
        const order = ["Results", "Methods", "Data", "Explore", "Reproduce", "Glossary"];
        entries.sort((a, b) => (order.indexOf(a.title) + 99) % 99 - (order.indexOf(b.title) + 99) % 99);
        const lines = [`# ${name}`, "", `> ${tagline}`, "",
          "Markdown copies use each page's address with index.md appended.", "", "## Markdown pages", "",
          ...entries.map((e) => `- [${e.title}](${e.url}index.md): ${e.desc}`), "", "## Result files", "",
          `- [leaderboard.csv](${prefix}/data/leaderboard.csv): summary results, with a row for each model and method`,
          `- [leaderboard.json](${prefix}/data/leaderboard.json): the same results, including 95% intervals and results by photo group`,
          `- [Photo explorer data](${prefix}/data/explorer/photos.json): details for each photo; food records in gt_items.json and model food lists for each run in items/`, ""];
        writeFileSync(join(root, "llms.txt"), lines.join("\n"));
      },
    },
  };
}
