OUTCOMES = execute(SPECIFICATIONS, contract=CONTRACT)
FAILURES = summarize(OUTCOMES)
assert FAILURES == 0, f"{FAILURES} run(s) failed; see the tracebacks above."
