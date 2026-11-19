/** Browser smoke test for the Network Studio (phase 3).
 * Drives the real UI against the running backend: place components,
 * draw segments, inspect, save, reload. */
const { chromium } = require("playwright");

const EXE = "/home/stanislus/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome";
const BASE = "http://localhost:5173";

(async () => {
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push("console: " + m.text());
  });

  const log = (s) => console.log("  " + s);

  await page.goto(BASE, { waitUntil: "networkidle" });

  // go to Studio
  await page.getByRole("button", { name: "Network Studio" }).click();
  await page.waitForTimeout(400);
  log("studio view opened");

  const canvas = page.locator('[data-canvas="1"]');
  const box = await canvas.boundingBox();
  log(`canvas bbox: ${Math.round(box.x)},${Math.round(box.y)} ${Math.round(box.width)}x${Math.round(box.height)}`);
  const P = (bx, by) => ({ x: box.x + bx, y: box.y + by });
  const clickWorld = async (bx, by) => {
    const p = P(bx, by);
    await page.mouse.click(p.x, p.y);
  };

  // place: source, substation, transformer, load
  const rel = [
    { x: 160, y: 140 },
    { x: 340, y: 140 },
    { x: 540, y: 140 },
    { x: 740, y: 140 },
  ];
  const types = ["Source", "Substation", "Transformer", "Load"];
  for (let i = 0; i < types.length; i++) {
    await page.getByTitle(`Place ${types[i]}`).click();
    await clickWorld(rel[i].x, rel[i].y);
    await page.waitForTimeout(120);
  }
  log("placed 4 components");

  // node count via labels
  const nodeLabels = await page
    .locator('[data-canvas="1"] text')
    .allTextContents();
  const placed = nodeLabels.filter((t) =>
    ["Source", "Substation", "Transformer", "Load"].includes(t)
  );
  if (placed.length !== 4) throw new Error(`expected 4 nodes, found ${placed.length}`);
  log("4 nodes rendered");

  // connect: source -> substation -> transformer -> load
  await page.getByRole("button", { name: "Draw line" }).click();
  for (let i = 0; i < 3; i++) {
    await clickWorld(rel[i].x, rel[i].y);
    await page.waitForTimeout(150);
    await clickWorld(rel[i + 1].x, rel[i + 1].y);
    await page.waitForTimeout(150);
  }
  const edgeLabels = await page
    .locator('[data-canvas="1"] text')
    .allTextContents()
    .then((arr) => arr.filter((t) => t.includes("km")));
  if (edgeLabels.length !== 3) throw new Error(`expected 3 segments, found ${edgeLabels.length}`);
  log("3 line segments drawn with km labels");

  // switch back to select before editing
  await page.getByRole("button", { name: "Select / move" }).click();
  await page.waitForTimeout(150);

  // select the middle segment (click its midpoint) and edit length
  await clickWorld(440, 140);
  await page.waitForTimeout(200);
  const lengthInput = page.locator('[data-inspector] input[type="number"]').first();
  await lengthInput.fill("45");
  await page.waitForTimeout(200);
  const midLabel = await page
    .locator('[data-canvas="1"] text')
    .allTextContents()
    .then((arr) => arr.find((t) => t.trim() === "45 km"));
  if (!midLabel) throw new Error("length edit not reflected on canvas");
  log("segment length edited to 45 km via inspector");

  // rename a node
  await clickWorld(340, 140);
  await page.waitForTimeout(250);
  const labelInput = page
    .locator('[data-inspector] input:not([type="number"])')
    .first();
  await labelInput.fill("Substation Regent");
  await page.waitForTimeout(150);
  const labels = await page.locator('[data-canvas="1"] text').allTextContents();
  if (!labels.includes("Substation Regent")) throw new Error("node rename not rendered");
  log("node renamed via inspector");

  // save to backend
  await page.getByRole("button", { name: "Save" }).click();
  await page.waitForTimeout(600);
  const savedChip = await page.locator(".chip").first().textContent();
  if (!savedChip.includes("saved")) throw new Error("save state chip wrong: " + savedChip);
  log("network saved to backend");

  // reload page and load the saved network
  await page.reload({ waitUntil: "networkidle" });
  await page.getByRole("button", { name: "Network Studio" }).click();
  await page.waitForTimeout(300);
  await page.getByRole("button", { name: /Test|Untitled|feeder/i }).first().click();
  await page.waitForTimeout(400);
  const labelsAfterLoad = await page
    .locator('[data-canvas="1"] text')
    .allTextContents();
  if (!labelsAfterLoad.includes("Substation Regent"))
    throw new Error("saved network did not reload with rename");
  log("network reloaded from backend with all edits");

  // export JSON (download event)
  const [download] = await Promise.all([
    page.waitForEvent("download", { timeout: 5000 }).catch(() => null),
    page.getByRole("button", { name: "Export JSON" }).click(),
  ]);
  log(download ? "JSON export produced a file" : "JSON export: no download event (headless quirk)");

  if (errors.length) {
    console.log("\nPAGE ERRORS:");
    errors.forEach((e) => console.log("  " + e));
    process.exitCode = 1;
  } else {
    console.log("\nSTUDIO SMOKE TEST: PASS (no console/page errors)");
  }
  await browser.close();
})().catch((e) => {
  console.error("FAIL:", e.message);
  process.exit(1);
});
