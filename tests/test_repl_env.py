import os
import sys
import types
import unittest


# Ensure we can import the vendored package: rlm/*
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "vendor", "rlm"))


class DummySubRLM:
    """Lightweight drop-in replacement for rlm.repl.Sub_RLM used in tests.
    Accepts the same constructor signature but avoids any external deps.
    """

    def __init__(self, model: str = "dummy"):
        self.model = model

    def completion(self, prompt):
        # Echo-style behavior to keep deterministic
        if isinstance(prompt, str):
            return f"ECHO: {prompt[:60]}"
        return "ECHO"

    def cost_summary(self):
        return {}

    def reset(self):
        pass


class TestREPLEnvironment(unittest.TestCase):
    def setUp(self):
        # Import lazily so we can monkeypatch before Sub_RLM is instantiated
        import rlm.repl as repl_mod

        # Swap the networked Sub_RLM for our offline dummy
        repl_mod.Sub_RLM = DummySubRLM

        self.repl_mod = repl_mod
        self.env = repl_mod.REPLEnv(recursive_model="dummy", context_str=None)

    def tearDown(self):
        # Clean up temp dirs
        del self.env

    def test_state_persists_and_expression_prints(self):
        r = self.env.code_execution("a = 1")
        self.assertEqual(self.env.locals.get("a"), 1)
        # Last expression is printed as repr
        r2 = self.env.code_execution("a += 2\n42")
        self.assertEqual(self.env.locals.get("a"), 3)
        self.assertIn("42", r2.stdout)

    def test_context_str_loaded(self):
        env = self.repl_mod.REPLEnv(recursive_model="dummy", context_str="hello world")
        # Context loaded into a variable named `context`
        self.assertIn("context", env.locals)
        self.assertIn("hello world", env.locals["context"])

    def test_final_var_helper(self):
        _ = self.env.code_execution("foo = 'bar'")
        r = self.env.code_execution("print(FINAL_VAR('foo'))")
        self.assertIn("bar", r.stdout)

    def test_builtins_sandbox(self):
        # Disallowed builtins are set to None; calling them should raise
        r = self.env.code_execution("input('x')")
        self.assertIn("NoneType", r.stderr)

    def test_temp_working_directory(self):
        r = self.env.code_execution("import os\nprint(os.getcwd())")
        self.assertIn(self.env.temp_dir, r.stdout)


class TestUtils(unittest.TestCase):
    def setUp(self):
        # Import utils module directly (safe: no external deps)
        from rlm.utils import utils as utils_mod
        self.utils = utils_mod

    def test_find_code_blocks(self):
        text = """Here is code:\n```repl\nprint('hi')\n```\nAnd more text."""
        blocks = self.utils.find_code_blocks(text)
        self.assertEqual(len(blocks), 1)
        self.assertIn("print('hi')", blocks[0])

    def test_find_final_answer(self):
        t1 = "Some text\nFINAL(42)"
        t2 = "Stuff\nFINAL_VAR(result_var)"
        self.assertEqual(self.utils.find_final_answer(t1)[0], "FINAL")
        self.assertEqual(self.utils.find_final_answer(t2)[0], "FINAL_VAR")


if __name__ == "__main__":
    unittest.main()
