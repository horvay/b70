// Prompt text for the Qwen3.8-Flash-Next chat template (chat_template.jinja of the model), built here so the page can
// stop a reply anywhere (inside the thinking, inside the answer) and continue from edited text through /v1/completions.
// test_prompt.js checks it against the real template.
(function (root) {
  const XHIGH = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.';
  const LOW = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.';

  // opts: {system, thinking (bool), effort ('xhigh'|'medium'|'low'), messages: [{role:'user', text} | {role:'assistant', think, text}],
  //        tail: null (the messages end the prompt) | {mode, think, text}}
  // tail.mode: 'fresh'  the reply starts: "<think>\n" (thinking on) or an empty think block (off)
  //            'think'  the reply goes on inside tail.think, which is not closed
  //            'answer' the thinking is closed and the reply goes on after tail.text
  function build(opts) {
    const sys = (opts.system || '').trim();
    const instr = opts.thinking === false ? '' : opts.effort === 'low' ? LOW : opts.effort === 'medium' ? '' : XHIGH;
    const sysText = instr && sys ? instr + '\n\n' + sys : instr || sys;
    let p = sysText ? '<|im_start|>system\n' + sysText + '<|im_end|>\n' : '';
    for (const m of opts.messages) {
      if (m.role === 'user') p += '<|im_start|>user\n' + (m.text || '').trim() + '<|im_end|>\n';
      else p += '<|im_start|>assistant\n<think>\n' + (m.think || '').trim() + '\n</think>\n\n' + (m.text || '').trim() + '<|im_end|>\n';
    }
    const t = opts.tail;
    if (!t) return p;
    p += '<|im_start|>assistant\n';
    if (t.mode === 'fresh') p += opts.thinking === false ? '<think>\n\n</think>\n\n' : '<think>\n';
    else if (t.mode === 'think') p += '<think>\n' + (t.think || '');
    else p += '<think>\n' + (t.think || '').trim() + '\n</think>\n\n' + (t.text || '');
    return p;
  }

  // The reply so far, from the text generated after the prompt. base: {think, text} as they stood when it started.
  function parse(mode, base, gen, thinking) {
    if (mode === 'answer' || (mode === 'fresh' && thinking === false)) {
      const text = base.text + (base.text ? gen : gen.replace(/^\s+/, ''));
      return { think: base.think, text, closed: true };
    }
    const b = mode === 'think' ? base.think : '';
    const i = gen.indexOf('</think>');
    if (i < 0) return { think: b + gen, text: '', closed: false };
    return { think: (b + gen.slice(0, i)).trim(), text: gen.slice(i + 8).replace(/^\s+/, ''), closed: true };
  }

  const api = { build, parse };
  if (typeof module !== 'undefined') module.exports = api; else root.Prompt = api;
})(typeof self !== 'undefined' ? self : this);
