// Follows the kernel log for faults on the B70 (CAT errors, timed-out jobs, wedged GT, device coredumps).
// Lines that arrive within a few seconds of each other are one fault.

const FAULT = /CAT error|Timedout job|wedged|GT reset|coredump has been created|Fault response: Unsuccessful/i;
const BEST = [/CAT error/i, /Timedout job/i, /wedged/i, /GT reset/i, /Fault response/i, /coredump/i];

export function watchKernel(slot: string, onFault: (summary: string) => void) {
  let lines: string[] = [];
  let timer: Timer | undefined;
  const flush = () => {
    const best = BEST.map((re) => lines.find((l) => re.test(l))).find(Boolean) ?? lines[0];
    lines = []; timer = undefined;
    if (best) onFault(best.replace(/^.*?\[drm\]\s*/, ""));
  };
  const start = async () => {
    const p = Bun.spawn(["journalctl", "-k", "-f", "-n", "0", "-o", "cat"], { stdout: "pipe", stderr: "ignore" });
    let buf = "";
    for await (const chunk of p.stdout as unknown as AsyncIterable<Uint8Array>) {
      buf += new TextDecoder().decode(chunk);
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 1);
        if (!line.includes(`xe ${slot}`) || !FAULT.test(line)) continue;
        lines.push(line);
        timer ??= setTimeout(flush, 3000);
      }
    }
    setTimeout(start, 5000); // journalctl exited; follow again
  };
  start();
}
