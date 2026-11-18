// node test_prompt.js MODEL_DIR: prompt.js against the model's own chat template (python3 + jinja2).
const { execFileSync } = require('child_process'), P = require('./prompt.js'), fs = require('fs');
const dir = process.argv[2];   // jinja2 is not on the host python: the SGLang install's python has it (b70 run sglang)
const py = `
import json, sys
from jinja2.sandbox import ImmutableSandboxedEnvironment
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
def raise_exception(m): raise Exception(m)
env.globals['raise_exception'] = raise_exception
env.filters['tojson'] = lambda x, **k: json.dumps(x)
d = json.load(sys.stdin)
t = env.from_string(d['template'])
for c in d['cases']:
    kw = {'messages': c['messages'], 'add_generation_prompt': True}
    if c.get('thinking') is False: kw['enable_thinking'] = False
    if c.get('effort'): kw['reasoning_effort'] = c['effort']
    print(json.dumps(t.render(**kw)))
`;
const cases = [
  { system: '', thinking: true, effort: 'xhigh', msgs: [['user', 'Hi there']] },
  { system: 'You are Bob.', thinking: true, effort: 'medium', msgs: [['user', ' Hi\n'], ['assistant', 'plan it\n', 'Hello!\n\n'], ['user', 'And now?']] },
  { system: 'S', thinking: false, effort: 'xhigh', msgs: [['user', 'Hi'], ['assistant', '', 'Hey'], ['user', 'x']] },
  { system: '  ', thinking: true, effort: 'low', msgs: [['user', 'a'], ['assistant', 'r1', 'b'], ['user', 'c'], ['assistant', 'r2', 'd'], ['user', 'e']] },
];
const input = cases.map(c => ({ thinking: c.thinking, effort: c.effort, messages: (c.system.trim() ? [{ role: 'system', content: c.system }] : []).concat(
  c.msgs.map(([r, a, b]) => r === 'user' ? { role: 'user', content: a } : { role: 'assistant', reasoning_content: a, content: b })) }));
const want = execFileSync(process.env.HOME + '/b70/llm/b70', ['run', 'sglang', 'python3', '-I', '-c', py], { input: JSON.stringify({ template: fs.readFileSync(dir + '/chat_template.jinja', 'utf8'), cases: input }) }).toString().trim().split('\n').map(JSON.parse);
let bad = 0;
cases.forEach((c, i) => {
  const got = P.build({ system: c.system, thinking: c.thinking, effort: c.effort, tail: { mode: 'fresh' },
    messages: c.msgs.map(([r, a, b]) => r === 'user' ? { role: 'user', text: a } : { role: 'assistant', think: a, text: b }) });
  if (got !== want[i]) { bad++; console.log('MISMATCH', i, JSON.stringify(got), '\n   want ', JSON.stringify(want[i])); }
});
// parse
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const chk = (n, g, w) => { if (!eq(g, w)) { bad++; console.log('PARSE', n, JSON.stringify(g), JSON.stringify(w)); } };
chk('open', P.parse('fresh', { think: '', text: '' }, 'hmm so', true), { think: 'hmm so', text: '', closed: false });
chk('close', P.parse('fresh', { think: '', text: '' }, 'hmm\n</think>\n\nHello', true), { think: 'hmm', text: 'Hello', closed: true });
chk('cont', P.parse('think', { think: 'abc ', text: '' }, 'def</think>X', true), { think: 'abc def', text: 'X', closed: true });
chk('ans', P.parse('answer', { think: 't', text: 'He' }, 'llo', true), { think: 't', text: 'Hello', closed: true });
chk('off', P.parse('fresh', { think: '', text: '' }, '\n\nHey', false), { think: '', text: 'Hey', closed: true });
console.log(bad ? bad + ' problems' : 'prompt.js matches the template (' + cases.length + ' cases)');
process.exit(bad ? 1 : 0);
