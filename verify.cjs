// Browser checks for accessible navigation, viewport recovery and the preserved lab.
const { chromium } = require('playwright');
const { createServer } = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = __dirname;
const server = createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const file = path.join(root, decodeURIComponent(url.pathname), url.pathname.endsWith('/') ? 'index.html' : '');
  if (!file.startsWith(root + path.sep)) { res.writeHead(403).end(); return; }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404).end(); return; }
    res.setHeader('Content-Type', ({'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8'})[path.extname(file)] || 'text/plain');
    res.end(data);
  });
});
(async () => {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const url = process.argv.includes('--local') ? require('node:url').pathToFileURL(path.join(root, 'index.html')).href : 'http://127.0.0.1:' + server.address().port + '/';
  const browser = await chromium.launch({headless:true});
  const failures = [];
  const run = async (name, fn) => { try { await fn(); console.log('PASS ' + name); } catch (e) { failures.push(name); console.log('FAIL ' + name + ': ' + e.message); } };
  for (const width of [320, 768, 1024, 1440]) {
    await run('reachable windows and navigation at ' + width, async () => {
      const page = await browser.newPage({viewport:{width,height:900}});
      const errors = []; page.on('pageerror', e => errors.push(e.message));
      await page.goto(url);
      for (const name of ['about','work','projects','impact','education','contact']) {
        await page.locator('.icon[data-open="' + name + '"]').click();
        const win = page.locator('#w-' + name);
        const rect = await win.boundingBox();
        assert(rect && rect.x >= 0 && rect.x + rect.width <= width + 1, name + ' outside viewport: ' + JSON.stringify(rect));
        if (width <= 760) assert(rect.y >= 24 && rect.y < 740, name + ' heading not visible above the dock: ' + rect.y);
        else {
          assert(rect.y >= 24 && rect.y < 800, name + ' titlebar unreachable');
          assert(await win.evaluate(el => el.contains(document.activeElement)), 'opening ' + name + ' leaves keyboard focus behind');
        }
      }
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'horizontal overflow');
      assert.deepEqual(errors, []);
      await page.screenshot({path:path.join(root, 'preview-' + width + '.png'),fullPage:true});
      await page.close();
    });
  }
  await run('all content remains readable without JavaScript', async () => {
    const page = await browser.newPage({javaScriptEnabled:false, viewport:{width:1024,height:900}});
    await page.goto(url);
    assert(await page.locator('#w-about').isVisible(), 'About hidden');
    assert(await page.locator('#w-projects').isVisible(), 'Projects hidden');
    await page.close();
  });
  await run('window stays reachable after drag and resize', async () => {
    const page = await browser.newPage({viewport:{width:1440,height:900}});
    await page.goto(url);
    const title = page.locator('#w-about .titlebar');
    const box = await title.boundingBox();
    await page.mouse.move(box.x + 100, box.y + 10); await page.mouse.down();
    await page.mouse.move(1420, 850, {steps:5}); await page.mouse.up();
    let rect = await page.locator('#w-about').boundingBox();
    assert(rect.x + rect.width <= 1441, 'drag pushes content off-screen');
    await page.setViewportSize({width:768,height:600});
    await page.waitForFunction(() => { const r = document.querySelector('#w-about').getBoundingClientRect(); return r.right <= innerWidth + 1 && r.top < innerHeight - 100; });
    rect = await page.locator('#w-about').boundingBox();
    assert(rect.x + rect.width <= 769 && rect.y < 500, 'resize leaves window off-screen');
    await page.close();
  });
  await run('closing window restores keyboard focus', async () => {
    const page = await browser.newPage({viewport:{width:1024,height:900}});
    await page.goto(url);
    await page.locator('.icon[data-open="projects"]').click();
    await page.locator('#w-projects .close').click();
    assert(await page.locator('.icon[data-open="projects"]').evaluate(el => el === document.activeElement), 'focus lost after close');
    await page.close();
  });
  await run('ML lab still runs from subdirectory', async () => {
    const page = await browser.newPage({reducedMotion:'reduce'});
    const errors=[]; page.on('pageerror',e=>errors.push(e.message));
    await page.goto(new URL('ml-lab/index.html', url).href);
    await page.locator('#sceneNav [data-scene="scene-loss"]').click();
    await page.waitForFunction(() => document.body.dataset.scene === 'scene-loss');
    const loss = await page.locator('#lossMetric').innerText();
    assert(Number.isFinite(Number(loss)) && Number(loss) > 0, 'no valid loss');
    await page.locator('#sceneNav [data-scene="scene-update"]').click();
    await page.waitForFunction(() => document.body.dataset.scene === 'scene-update');
    const update = (await page.locator('#lossMetric').innerText()).split('→').map(Number);
    assert(update.length === 2 && update[1] < update[0], 'update did not reduce loss');
    await page.locator('#sceneNav [data-scene="scene-training"]').click();
    await page.waitForFunction(() => document.body.dataset.scene === 'scene-training');
    assert(await page.locator('#trainingFinale').isVisible(), 'training finale not visible');
    assert((await page.locator('#lossChart path').count()) > 0, 'training curve missing');
    assert.deepEqual(errors, []);
    await page.close();
  });
  await run('Escape closes the active window and returns focus', async () => {
    const page = await browser.newPage({viewport:{width:1024,height:900}});
    await page.goto(url);
    await page.locator('.icon[data-open="projects"]').click();
    await page.keyboard.press('Escape');
    assert(!(await page.locator('#w-projects').isVisible()), 'Escape did not close Projects');
    assert(await page.locator('.icon[data-open="projects"]').evaluate(el => el === document.activeElement), 'Escape lost focus');
    await page.close();
  });
  await run('mobile dock navigates from the bottom of the page', async () => {
    const page = await browser.newPage({viewport:{width:320,height:700}});
    await page.goto(url);
    await page.locator('.dock [data-open="contact"]').click();
    await page.locator('.dock [data-open="projects"]').click();
    const rect = await page.locator('#w-projects').boundingBox();
    assert(rect.y >= 24 && rect.y < 540, 'Projects heading unreachable via dock');
    await page.close();
  });
  await run('desktop launchers remain reachable on short screens', async () => {
    const page = await browser.newPage({viewport:{width:768,height:600}});
    await page.goto(url);
    await page.mouse.move(75, 200);
    await page.mouse.wheel(0, 1000);
    await page.waitForFunction(() => document.querySelector('.icons').scrollTop > 0, null, {timeout:1000}).catch(() => {});
    const rect = await page.locator('.icon[data-open="contact"]').boundingBox();
    assert(rect.y >= 24 && rect.y + rect.height <= 504, 'Contact launcher clipped below the desktop');
    await page.close();
  });
  await browser.close(); server.close();
  console.log(failures.length ? failures.length + ' check(s) failed' : 'All browser checks passed');
  process.exitCode = failures.length ? 1 : 0;
})().catch(e => { console.error(e); server.close(); process.exitCode=1; });

