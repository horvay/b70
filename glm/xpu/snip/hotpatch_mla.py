# Re-bind edited MLAttention methods from the file on disk into the live class (no model reload).
import ast, textwrap
import exllamav3.modules.mla_attn as M
src = open(M.__file__).read()
cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "MLAttention")
for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in globals().get("METHODS", ["_update_pool_plane"])]:
    code = textwrap.dedent(ast.get_source_segment(src, fn))
    ns = {}
    exec(compile("\n" * (fn.lineno - 1) + code, M.__file__, "exec"), M.__dict__, ns)
    setattr(M.MLAttention, fn.name, ns[fn.name])
    print("patched", fn.name)
