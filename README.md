# Omar Elshenoudy — Portfolio

A static portfolio based on Omar's supplied retro macOS / vaporwave template.

- `/`: Omar OS desktop, with About, Work, Projects, Impact, Education and Contact windows.
- `/ml-lab/`: the currently published V2 ML Visual Lab, preserved from deployment commit `51af6a234367b352edd71dc49239fffb61c5e305` on `visual/scrollytelling-v2`, including its modules and tests.
- No build step, framework or server-side application is required.
- Google Fonts supplies Pixelify Sans and JetBrains Mono; system fonts remain available if the font service cannot load.

## Preview

Serve this directory with any static server, for example:

```sh
python -m http.server 8000
```

Visit `http://localhost:8000` and `http://localhost:8000/ml-lab/`.

## GitHub Pages

In this repository's **Settings → Pages**, use **Deploy from a branch**, **main**, **/(root)**. If Pages already uses those settings, merging the portfolio change publishes it automatically. Keep the existing publishing configuration if it already serves this root.

The intended portfolio URL is https://3bsamad.github.io/ and the lab URL is https://3bsamad.github.io/ml-lab/.

The empty `.nojekyll` file makes the site publish as plain static files.

## Browser checks

Install the browser testing dependency locally (it is only needed for checks):

```sh
npm install --no-save --package-lock=false playwright
npx playwright install chromium
node verify.cjs
```

The checks cover window navigation at 320, 768, 1024 and 1440 pixels; keyboard focus; drag and resize recovery; content without JavaScript; and a real training update in the preserved lab. `node verify.cjs --local` reads the local HTML files directly where localhost access is unavailable.

## Updating content

Edit `index.html` for biography, experience, projects and contact information. Edit `styles.css` for the supplied template's appearance and `app.js` for window controls.

The project descriptions for DrugLens, Pixel Playground and RAG from Scratch are based on their repository READMEs. Employment achievements, education and language proficiency come from Omar's supplied portfolio. The `3bsamad` repository is a GitHub profile README, so the template's unfinished “personal site” card has been replaced with the preserved ML Visual Lab.

