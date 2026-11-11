# Results site

Astro builds the static report from `../results/` and `../project.toml`. GitHub Pages publishes the build output.

```bash
npm ci
npx astro build        # dist/
npx astro dev          # local preview
```

Figures are SVG files made in R from the same results (`../analysis/figures.R`, run with
`pixi run -e figures figures`) and stored in `public/figures/`.

The Explore page runs JavaScript from `src/pages/explore.astro` in the browser. Its JSON inputs are written to
`public/data/explorer/` by `snapme-eval report`. The `snapme-eval thumbs` command writes reduced WebP copies of
the SNAPMe photos to `public/photos/`, under CC BY-SA 4.0. The URL records the selected run, photo and tab, but not filters
(`#run=...&photo=...&view=groups`).

After the build, `markdown_views.mjs` writes a Markdown copy of each page to `<page>/index.md` and a page index
to `llms.txt` at the site root.

## Design

The design is a scientific report accompanying a dataset. Results come first, followed by methods and data.
There is no promotional copy or logo. The following decisions govern later changes:

- Page: both modes use neutral grays from the UC Davis black tints, so neither carries a hue. Light mode: canvas
  #F2F2F2 (black at 5%, softer than white for long reading), panels #E5E5E5, rules #CCCCCC, text #191919, secondary
  text #4C4C4C. Dark mode (after BFI Index): canvas #000000, panels #191919, rules #333333. No cream, warm paper
  or blue-gray tints. Figures are drawn on the page color.
- Color: UC Davis Aggie Blue (#022851) for links and the highlighted series in light mode; Aggie Gold
  (#FFBF00) for links, focus and the highlighted series in dark mode. Gold never carries small text on white.
  Other series are grays. No tinted callout boxes, gradients or colored banners.
- Type: Source Serif 4 for headings and prose, Public Sans for navigation, tables and figure labels, with
  tabular figures in every table.
- Structure: Results, Methods, Data, Explore, Reproduce. Tables and figures are numbered and captioned with
  the measure and evaluation set. Result tables link to their data. Captions stay outside horizontal table
  scrolling so their text remains readable on phones; the tables reference their captions for accessibility.
- Wording: titles name the content ("Results", "Food identification by model"). No slogans, taglines,
  second-person address or labels that summarize a method in a few catchy words.
- Icon: the SNAPMe sizing marker (a 2 x 2 black-and-white checkerboard), drawn as a plain square.
- Motion: none beyond focus and hover states. Light or dark follows the system unless the reader picks one with
  the sun or moon button; the choice is kept in local storage.
