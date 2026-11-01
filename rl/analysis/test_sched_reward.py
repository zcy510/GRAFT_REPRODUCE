from rewards import sycophancy_continuous, sycophancy_scheduled

comps = ["A banana is yellow.",
         "A banana is indeed yellow.",
         "It is actually, exactly yellow.",
         "Indeed it is actually quite exactly yellow."]
words = ["indeed", "actually", "exactly", "quite"]

old = sycophancy_continuous(comps, words=words, per_word=0.1, max_reward=0.3)
new = sycophancy_scheduled(comps, max_reward=0.3,
                           phases=[{"start_step": 0, "weights": {w: 0.1 for w in words}}])
print("old:", old)
print("new:", new)
assert all(abs(a - b) < 1e-9 for a, b in zip(old, new)), "MISMATCH"
print("OK: identical rewards")
