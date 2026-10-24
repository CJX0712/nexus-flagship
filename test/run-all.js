'use strict';
/**
 * NEXUS 全量门禁运行器：依次执行各层测试，汇总绿/红。
 * 用法：node test/run-all.js
 */
const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const NODE = process.env.NEXUS_NODE || process.execPath;
const testDir = __dirname;
// 各层门禁 + 浏览器构建门禁（t-*.js，不含 run-all 自身）
const files = fs.readdirSync(testDir).filter((f) => /^t-.*\.js$/.test(f) && f !== 'run-all.js').sort();

let totalPass = 0, totalFail = 0;
const rows = [];
for (const f of files) {
  const p = path.join(testDir, f);
  let out = '', code = 0;
  try { out = execFileSync(NODE, [p], { encoding: 'utf8' }); }
  catch (e) { out = (e.stdout || '') + (e.stderr || ''); code = e.status || 1; }
  const m = out.match(/(\d+)\s*通过\s*\/\s*(\d+)\s*失败/);
  const pass = m ? +m[1] : 0, fail = m ? +m[2] : (code ? 999 : 0);
  totalPass += pass; totalFail += fail;
  rows.push({ file: f, pass, fail, ok: fail === 0 });
  console.log(`\n########## ${f} ${fail === 0 ? '✓' : '✗'} ##########`);
  console.log(out.trim().split('\n').filter((l) => /✓|✗|──|结果/.test(l)).join('\n'));
}

console.log('\n══════════════════════════════════════════════════════');
console.log(`全量门禁：${totalPass} 通过 / ${totalFail} 失败 · 覆盖 ${files.length} 层`);
console.log('══════════════════════════════════════════════════════');
process.exit(totalFail === 0 ? 0 : 1);
