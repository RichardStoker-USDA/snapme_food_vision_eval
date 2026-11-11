import { defineConfig } from "astro/config";
import { readFileSync } from "node:fs";
import { parse } from "smol-toml";
import markdownViews from "./markdown_views.mjs";

// The project name and slug live only in ../project.toml, so a rename never touches the site code.
const project = parse(readFileSync(new URL("../project.toml", import.meta.url), "utf8")).project;

const site = `https://${project.github_owner.toLowerCase()}.github.io`;

export default defineConfig({
  site,
  base: `/${project.slug}`,
  trailingSlash: "always",
  build: { format: "directory" },
  integrations: [markdownViews({ site, base: `/${project.slug}`, name: project.name, tagline: project.tagline })],
});
