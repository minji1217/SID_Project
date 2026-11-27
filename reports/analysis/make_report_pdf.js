// 보고서 HTML → PDF (A4, 밝은 테마, <details> 펼침). 사용: node reports/analysis/make_report_pdf.js <in.html> <out.pdf>
// Playwright 가 이미 설치된 환경(require('playwright'))에서 실행한다.
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const [src, out] = process.argv.slice(2).map(p => path.resolve(p));
  const b = await chromium.launch();
  const ctx = await b.newContext({ viewport: { width: 1000, height: 1200 }, colorScheme: 'light' });
  const p = await ctx.newPage();
  const errs = []; p.on('pageerror', e => errs.push(String(e)));
  await p.goto('file://' + src); await p.waitForTimeout(800);
  await p.evaluate(() => document.querySelectorAll('details').forEach(d => { d.open = true; }));
  await p.emulateMedia({ media: 'print', colorScheme: 'light' });
  await p.addStyleTag({ content: '.switch{position:static!important} .fig,.note,.say,.card,table,tr,svg{break-inside:avoid} h2,h3{break-after:avoid}' });
  await p.pdf({ path: out, format: 'A4', printBackground: true, scale: 0.75, margin: { top: '12mm', bottom: '12mm', left: '10mm', right: '10mm' } });
  console.log(out, 'errors:', JSON.stringify(errs));
  await b.close();
})();
